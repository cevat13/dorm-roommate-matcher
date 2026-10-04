"""Command line interface.

::

    roommate seed --count 400          # öğrenci popülasyonu üret
    roommate assign                    # optimal yerleşimi hesapla
    roommate rooms                     # son planın odaları
    roommate room 12                   # bir odanın gerekçesi
    roommate candidates 5              # bir öğrenci için uyumlu adaylar
    roommate compare 5 12              # iki öğrencinin uyum dökümü
    roommate evaluate                  # metrik ve strateji karşılaştırması
    roommate stats                     # popülasyon dağılımları
    roommate create-admin              # yönetici hesabı
    roommate serve                     # API
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from roommate_matcher import __version__
from roommate_matcher.config import get_settings
from roommate_matcher.data.database import SqlStudentRepository
from roommate_matcher.data.generator import generate_students
from roommate_matcher.data.repository import write_students
from roommate_matcher.domain.enums import MatchVerdict
from roommate_matcher.domain.models import AssignmentPlan, FeatureContribution
from roommate_matcher.exceptions import RoommateMatcherError
from roommate_matcher.logging_config import configure_logging
from roommate_matcher.matching.assignment import (
    RoomAssigner,
    available_strategies,
    get_strategy,
)
from roommate_matcher.matching.constraints import describe_violations
from roommate_matcher.matching.engine import CompatibilityEngine
from roommate_matcher.matching.evaluation import compare_metrics, evaluate_plans
from roommate_matcher.matching.explain import impact_table
from roommate_matcher.matching.metrics import available_metrics, get_metric

app = typer.Typer(
    name="roommate",
    help="Yurt oda arkadaşı eşleştirme ve analiz aracı.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

VERDICT_STYLE = {
    MatchVerdict.MATCH: "green",
    MatchVerdict.PARTIAL: "yellow",
    MatchVerdict.CLASH: "red",
}


def _repository() -> SqlStudentRepository:
    """Open the configured repository.

    Returns:
        A ready :class:`SqlStudentRepository`.
    """
    return SqlStudentRepository()


def _require_students(repository: SqlStudentRepository) -> None:
    """Abort with a helpful message when the database is empty.

    Args:
        repository: The repository to check.

    Raises:
        typer.Exit: When there is no data to work with.
    """
    if repository.count() == 0:
        console.print("[red]Veritabanı boş.[/red] Önce: [bold]roommate seed[/bold]")
        raise typer.Exit(code=1)


def _score_colour(percentage: int) -> str:
    """Pick a colour for a compatibility percentage.

    Args:
        percentage: The score as an integer percentage.

    Returns:
        A Rich colour name.
    """
    if percentage >= 75:
        return "green"
    return "yellow" if percentage >= 55 else "red"


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Ayrıntılı log")] = False,
) -> None:
    """Configure logging before any command runs.

    Args:
        verbose: Whether to enable debug logging.
    """
    configure_logging("DEBUG" if verbose else "WARNING")


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"dorm-roommate-matcher [bold cyan]v{__version__}[/bold cyan]")


@app.command()
def seed(
    count: Annotated[int, typer.Option(help="Üretilecek öğrenci sayısı")] = 400,
    seed_value: Annotated[int, typer.Option("--seed", help="Rastgelelik tohumu")] = 42,
    csv_out: Annotated[bool, typer.Option("--csv/--no-csv", help="CSV kopyası da yaz")] = True,
    reset: Annotated[
        bool, typer.Option("--reset", help="Mevcut öğrencileri ve planları sil")
    ] = False,
) -> None:
    """Fill the database with a synthetic student population.

    Args:
        count: How many students to generate.
        seed_value: Random seed for reproducibility.
        csv_out: Whether to also write a CSV snapshot.
        reset: Whether to clear existing students and plans first.

    Raises:
        typer.Exit: When the database already holds students and --reset was not given.
    """
    settings = get_settings()
    settings.ensure_directories()
    repository = _repository()

    existing = repository.count()
    if existing and not reset:
        console.print(
            f"[red]Veritabanında zaten {existing} öğrenci var.[/red] "
            "Üzerine yazmak için: [bold]roommate seed --reset[/bold]"
        )
        raise typer.Exit(code=1)
    if reset and existing:
        removed = repository.reset()
        console.print(f"[yellow]{removed} öğrenci ve kayıtlı planlar silindi.[/yellow]")

    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"), console=console, transient=True
    ) as progress:
        progress.add_task(f"{count} öğrenci üretiliyor...", total=None)
        students = generate_students(count, seed=seed_value)

        progress.add_task("Veritabanına yazılıyor...", total=None)
        repository.bulk_add(students)

    if csv_out:
        path = write_students(settings.generated_data_dir / "students.csv", students)
        console.print(f"CSV: [dim]{path}[/dim]")

    console.print(
        Panel(
            f"[green]{len(students)}[/green] öğrenci yüklendi "
            f"(toplam: [bold]{repository.count()}[/bold])",
            title="Hazır",
        )
    )


@app.command()
def assign(
    strategy: Annotated[str, typer.Option("--strategy", "-s", help="Yerleşim stratejisi")] = "",
    metric: Annotated[str, typer.Option("--metric", "-m", help="Uyum metriği")] = "",
    limit: Annotated[int, typer.Option(help="Kaç öğrenci yerleştirilsin (0 = tümü)")] = 0,
    save: Annotated[bool, typer.Option("--save/--no-save", help="Planı kaydet")] = True,
    show: Annotated[int, typer.Option(help="Kaç oda gösterilsin")] = 10,
) -> None:
    """Compute a room plan for the whole population.

    Args:
        strategy: Strategy key; empty means the configured default.
        metric: Metric key; empty means the configured default.
        limit: Cap the population size, for quick experiments.
        save: Whether to persist the plan.
        show: How many rooms to print.
    """
    settings = get_settings()
    repository = _repository()
    _require_students(repository)

    students = repository.list_all()
    if limit:
        students = students[:limit]

    try:
        engine = CompatibilityEngine(metric or settings.default_metric)
        assigner = RoomAssigner(strategy or settings.default_strategy, engine)
    except RoommateMatcherError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    with console.status(f"{len(students)} öğrenci yerleştiriliyor..."):
        plan = assigner.assign(students)

    _print_plan_summary(plan, len(students))
    _print_rooms(plan, limit=show)

    if save:
        plan_id = repository.save_plan(plan, engine.metric.key)
        console.print(f"[dim]Plan kaydedildi (id={plan_id}).[/dim]")


def _print_plan_summary(plan: AssignmentPlan, population: int) -> None:
    """Print the headline figures of a plan.

    Args:
        plan: The plan to summarise.
        population: How many students it covers.
    """
    optimal_note = "[green]kesin optimal[/green]" if plan.optimal else "[yellow]sezgisel[/yellow]"
    lines = [
        f"Strateji: [bold]{plan.strategy}[/bold] ({optimal_note})",
        f"Öğrenci: {population} · Oda: [bold]{plan.room_count}[/bold] · "
        f"Yerleşmeyen: {len(plan.unplaced)}",
        f"Toplam uyum: [bold]{plan.total_score:.4f}[/bold] · "
        f"Ortalama: {plan.mean_score:.4f} · En kötü oda: {plan.min_score:.4f}",
        f"Süre: {plan.seconds:.2f} sn",
    ]
    if plan.violations:
        lines.append(f"[red]Kaçınılamayan ihlaller:[/red] {describe_violations(plan.violations)}")
    else:
        lines.append("[green]Hiçbir kısıt ihlali yok.[/green]")
    if plan.blocking_pairs:
        lines.append(
            f"[dim]Kararlılık: {len(plan.blocking_pairs)} blocking pair "
            f"(toplam uyum hedefi kararlılığı garanti etmez).[/dim]"
        )
    else:
        lines.append("[green]Kararlı: hiçbir blocking pair yok.[/green]")
    if plan.unplaced:
        lines.append(f"[yellow]Yerleşmeyen öğrenci: {list(plan.unplaced)}[/yellow]")
    console.print(Panel("\n".join(lines), title="Yerleşim planı"))


def _print_rooms(plan: AssignmentPlan, *, limit: int) -> None:
    """Print the best and worst rooms of a plan.

    Args:
        plan: The plan to print.
        limit: How many rooms to show.
    """
    if not plan.rooms:
        return
    table = Table(title=f"Odalar (en uyumlu {min(limit, plan.room_count)})")
    table.add_column("Oda", justify="right", style="dim")
    table.add_column("Öğrenci A")
    table.add_column("Öğrenci B")
    table.add_column("Uyum", justify="right")
    table.add_column("Dikkat", style="red")

    for room in plan.rooms[:limit]:
        pair = room.pair
        colour = _score_colour(pair.percentage)
        concerns = ", ".join(c.label for c in pair.concerns(2)) or "-"
        table.add_row(
            str(room.room_no),
            f"#{pair.first_id} {pair.first_name}",
            f"#{pair.second_id} {pair.second_name}",
            f"[{colour}]%{pair.percentage}[/{colour}]",
            concerns,
        )
    console.print(table)

    worst = plan.worst_rooms(3)
    if worst and plan.room_count > limit:
        table = Table(title="En düşük uyumlu odalar")
        table.add_column("Oda", justify="right", style="dim")
        table.add_column("Öğrenciler")
        table.add_column("Uyum", justify="right")
        table.add_column("Dikkat", style="red")
        for room in worst:
            pair = room.pair
            table.add_row(
                str(room.room_no),
                f"#{pair.first_id} + #{pair.second_id}",
                f"[{_score_colour(pair.percentage)}]%{pair.percentage}[/]",
                ", ".join(c.label for c in pair.concerns(2)) or "-",
            )
        console.print(table)


@app.command()
def rooms(
    worst: Annotated[bool, typer.Option("--worst", help="En düşük uyumluları göster")] = False,
    limit: Annotated[int, typer.Option(help="Kaç oda")] = 20,
) -> None:
    """List the rooms of the most recently saved plan.

    Args:
        worst: Whether to order by lowest compatibility instead of highest.
        limit: How many rooms to show.
    """
    repository = _repository()
    row = repository.latest_plan_row()
    if row is None:
        console.print("[red]Kayıtlı plan yok.[/red] Önce: [bold]roommate assign[/bold]")
        raise typer.Exit(code=1)

    stored = repository.plan_rooms(int(row.id))
    ordered = sorted(stored, key=lambda r: r.score, reverse=not worst)[:limit]
    by_id = {s.student_id: s for s in repository.list_all()}

    table = Table(
        title=(
            f"Plan #{row.id} · {row.strategy} · {row.room_count} oda · toplam {row.total_score:.4f}"
        )
    )
    table.add_column("Oda", justify="right", style="dim")
    table.add_column("Öğrenci A")
    table.add_column("Öğrenci B")
    table.add_column("Uyum", justify="right")
    table.add_column("İhlal", style="red")

    for stored_room in ordered:
        percentage = round(stored_room.score * 100)
        first = by_id.get(stored_room.first_id)
        second = by_id.get(stored_room.second_id)
        table.add_row(
            str(stored_room.room_no),
            f"#{stored_room.first_id} {first.display_name if first else '?'}",
            f"#{stored_room.second_id} {second.display_name if second else '?'}",
            f"[{_score_colour(percentage)}]%{percentage}[/]",
            stored_room.violations or "-",
        )
    console.print(table)
    if row.unplaced:
        console.print(f"[yellow]Yerleşmeyen: {row.unplaced}[/yellow]")


@app.command()
def room(
    room_no: Annotated[int, typer.Argument(help="Oda numarası")],
) -> None:
    """Explain why one room holds those two students.

    Args:
        room_no: The room to explain.
    """
    repository = _repository()
    plan_row = repository.latest_plan_row()
    if plan_row is None:
        console.print("[red]Kayıtlı plan yok.[/red] Önce: [bold]roommate assign[/bold]")
        raise typer.Exit(code=1)

    stored = {r.room_no: r for r in repository.plan_rooms(int(plan_row.id))}
    if room_no not in stored:
        console.print(f"[red]Oda {room_no} bu planda yok.[/red]")
        raise typer.Exit(code=1)

    target = stored[room_no]
    try:
        first = repository.get(target.first_id)
        second = repository.get(target.second_id)
    except RoommateMatcherError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    engine = CompatibilityEngine(str(plan_row.metric))
    pair = engine.explain_pair(first, second)

    console.print(Panel(first.summary(), title=f"#{first.student_id}", style="cyan"))
    console.print(Panel(second.summary(), title=f"#{second.student_id}", style="magenta"))
    console.print(
        Panel(
            pair.summary,
            title=f"[{_score_colour(pair.percentage)}]Oda {room_no} · uyum %{pair.percentage}[/]",
        )
    )
    _print_breakdown(pair.contributions, f"Oda {room_no}")


def _print_breakdown(contributions: Sequence[FeatureContribution], title: str) -> None:
    """Print the per-dimension impact table for one pairing.

    Args:
        contributions: The contributions behind the score.
        title: Table title.
    """
    table = Table(title=f"{title} · özellik dökümü")
    table.add_column("Özellik")
    table.add_column("Öğrenci A")
    table.add_column("Öğrenci B")
    table.add_column("Benzerlik", justify="right")
    table.add_column("Ağırlık", justify="right")
    table.add_column("Skora katkı", justify="right")
    table.add_column("Durum")

    for row in impact_table(contributions):
        style = VERDICT_STYLE[row.verdict]
        table.add_row(
            row.label,
            row.first_value,
            row.second_value,
            f"{row.similarity:.2f}",
            f"{row.weight:.1f}",
            f"{row.share_of_score * 100:.1f}%",
            f"[{style}]{row.verdict.label_tr}[/{style}]",
        )
    console.print(table)


@app.command()
def candidates(
    student_id: Annotated[int, typer.Argument(help="Öğrenci kimliği")],
    top: Annotated[int, typer.Option("--top", "-n", help="Kaç aday")] = 5,
    metric: Annotated[str, typer.Option("--metric", "-m")] = "",
    detail: Annotated[bool, typer.Option("--detail", "-d", help="Özellik dökümü")] = False,
) -> None:
    """Show the most compatible roommates for one student.

    An analysis tool: it does not place anybody, it answers who would suit this
    student best.

    Args:
        student_id: The student to analyse.
        top: How many candidates to show.
        metric: Metric key; empty means the configured default.
        detail: Whether to print the breakdown of the best candidate.
    """
    repository = _repository()
    _require_students(repository)
    try:
        student = repository.get(student_id)
        engine = CompatibilityEngine(metric or get_settings().default_metric)
        report = engine.candidates_for(student, repository.others(student_id), top_n=top)
    except RoommateMatcherError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(Panel(student.summary(), title=f"#{student.student_id}", style="cyan"))

    if not report.results:
        console.print("[yellow]Kısıtlara uyan aday bulunamadı.[/yellow]")
        console.print(f"[dim]Elenenler -> {describe_violations(report.rejections)}[/dim]")
        raise typer.Exit(code=0)

    table = Table(title=f"En uyumlu {len(report.results)} aday · metrik: {report.metric}")
    table.add_column("#", justify="right", style="dim")
    table.add_column("ID", justify="right")
    table.add_column("İsim")
    table.add_column("Uyum", justify="right")
    table.add_column("Öne çıkanlar")
    table.add_column("Dikkat", style="red")

    for rank, pair in enumerate(report.results, start=1):
        table.add_row(
            str(rank),
            str(pair.second_id),
            pair.second_name,
            f"[{_score_colour(pair.percentage)}]%{pair.percentage}[/]",
            ", ".join(c.label for c in pair.strengths(2)) or "-",
            ", ".join(c.label for c in pair.concerns(2)) or "-",
        )
    console.print(table)
    console.print(
        f"[dim]{report.considered} aday tarandı, {report.filtered_out} tanesi elendi. "
        f"{describe_violations(report.rejections)}[/dim]"
    )

    if detail and report.best:
        _print_breakdown(report.best.contributions, report.best.second_name)


@app.command()
def compare(
    first_id: Annotated[int, typer.Argument(help="Birinci öğrenci")],
    second_id: Annotated[int, typer.Argument(help="İkinci öğrenci")],
    metric: Annotated[str, typer.Option("--metric", "-m")] = "",
) -> None:
    """Explain the compatibility of two specific students.

    Args:
        first_id: The first student.
        second_id: The second student.
        metric: Metric key; empty means the configured default.
    """
    repository = _repository()
    _require_students(repository)
    try:
        first = repository.get(first_id)
        second = repository.get(second_id)
    except RoommateMatcherError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    engine = CompatibilityEngine(metric or get_settings().default_metric)
    pair = engine.explain_pair(first, second)

    console.print(Panel(first.summary(), title=f"#{first.student_id}", style="cyan"))
    console.print(Panel(second.summary(), title=f"#{second.student_id}", style="magenta"))
    console.print(
        Panel(pair.summary, title=f"[{_score_colour(pair.percentage)}]Uyum %{pair.percentage}[/]")
    )
    _print_breakdown(pair.contributions, second.display_name)


@app.command()
def evaluate(
    sample: Annotated[int, typer.Option(help="Metrik ölçümü için sorgu sayısı")] = 100,
    limit: Annotated[int, typer.Option(help="Strateji ölçümü için öğrenci sayısı")] = 200,
    save: Annotated[Path | None, typer.Option(help="Sonucu JSON olarak kaydet")] = None,
) -> None:
    """Benchmark the metrics and the assignment strategies.

    Args:
        sample: How many students to use as ranking queries.
        limit: How many students to use for the assignment comparison.
        save: Optional path to write both tables as JSON.
    """
    repository = _repository()
    _require_students(repository)
    students = repository.list_all()

    with console.status(f"{len(available_metrics())} metrik değerlendiriliyor..."):
        metric_report = compare_metrics(students, sample=sample)

    table = Table(title="Metrik kalitesi (kural tabanlı oracle'a göre)")
    for column in ("metric", "P@5", "P@10", "R@10", "MRR", "nDCG@10", "queries", "sec"):
        table.add_column(column, justify="left" if column == "metric" else "right")
    for row in metric_report.rows():
        table.add_row(*(str(row[key]) for key in row))
    console.print(table)
    if metric_report.winner:
        console.print(
            f"[green]En iyi metrik: {metric_report.winner.metric} "
            f"(nDCG@10 = {metric_report.winner.ndcg_at_10:.4f})[/green]"
        )

    subset = students[:limit]
    with console.status(f"{len(available_strategies())} strateji, {len(subset)} öğrenci..."):
        plan_scores = evaluate_plans(subset)

    table = Table(title=f"Yerleşim kalitesi · {len(subset)} öğrenci")
    columns = list(plan_scores[0].as_row()) if plan_scores else []
    for column in columns:
        table.add_column(column, justify="left" if column == "strateji" else "right")
    for scores in plan_scores:
        row = scores.as_row()
        table.add_row(*(str(row[key]) for key in columns))
    console.print(table)

    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metrics": metric_report.rows(),
            "strategies": [s.as_row() for s in plan_scores],
        }
        save.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"[dim]Kaydedildi: {save}[/dim]")


@app.command()
def strategies() -> None:
    """List every registered assignment strategy."""
    table = Table(title="Yerleşim stratejileri")
    table.add_column("Anahtar", style="cyan")
    table.add_column("Ad")
    table.add_column("Açıklama")
    table.add_column("Optimal", justify="center")
    for key in available_strategies():
        strategy = get_strategy(key)
        table.add_row(
            strategy.key,
            strategy.label,
            strategy.description,
            "✓" if strategy.optimal else "-",
        )
    console.print(table)


@app.command()
def metrics() -> None:
    """List every registered compatibility metric."""
    table = Table(title="Uyum metrikleri")
    table.add_column("Anahtar", style="cyan")
    table.add_column("Ad")
    table.add_column("Açıklama")
    table.add_column("Açıklanabilir", justify="center")
    for key in available_metrics():
        metric = get_metric(key)
        table.add_row(
            metric.key, metric.label, metric.description, "✓" if metric.explainable else "-"
        )
    console.print(table)


@app.command()
def stats() -> None:
    """Show population statistics."""
    from collections import Counter

    repository = _repository()
    _require_students(repository)
    students = repository.list_all()

    table = Table(title=f"Popülasyon · {len(students)} öğrenci")
    table.add_column("Dağılım")
    table.add_column("Değerler")
    for label, values in (
        ("Sınıf", Counter(f"{s.study_year}. sınıf" for s in students)),
        ("Bölüm", Counter(s.department.label_tr for s in students)),
        ("Uyku düzeni", Counter(s.sleep_schedule.label_tr for s in students)),
        ("Ders çalışma yeri", Counter(s.study_location.label_tr for s in students)),
        ("Sigara", Counter(s.smoking.label_tr for s in students)),
        ("Temizlik", Counter(str(s.cleanliness) for s in students)),
    ):
        rendered = ", ".join(f"{k}: {v}" for k, v in values.most_common(6))
        table.add_row(label, rendered)
    console.print(table)


@app.command(name="create-admin")
def create_admin(
    email: Annotated[str, typer.Option(prompt=True, help="Yönetici e-postası")],
    password: Annotated[
        str, typer.Option(prompt=True, hide_input=True, help="Parola (en az 8 karakter)")
    ],
) -> None:
    """Create a dormitory staff account for the API.

    Args:
        email: The login address.
        password: The password; at least 8 characters.
    """
    from roommate_matcher.api.security import hash_password

    if len(password) < 8:
        console.print("[red]Parola en az 8 karakter olmalı.[/red]")
        raise typer.Exit(code=1)

    repository = _repository()
    try:
        repository.add_admin(email.strip().lower(), hash_password(password))
    except RoommateMatcherError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Yönetici hesabı oluşturuldu: {email}[/green]")


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Dinlenecek adres")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port")] = 8000,
    reload: Annotated[bool, typer.Option("--reload", help="Otomatik yeniden yükle")] = False,
) -> None:
    """Start the HTTP API.

    Args:
        host: Bind address.
        port: Bind port.
        reload: Whether to enable auto-reload for development.
    """
    import uvicorn

    console.print(f"[green]API: http://{host}:{port}/docs[/green]")
    uvicorn.run("roommate_matcher.api.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
