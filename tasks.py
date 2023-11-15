from invoke import Context, task


@task
def autoformat(ctx: Context) -> None:
    print(f"Running auto-formatters...")
    ctx.run(f"isort .", pty=True, echo=True)
    ctx.run("black -l 120 .", pty=True, echo=True)
    print(f"Finished running auto-formatters.")
