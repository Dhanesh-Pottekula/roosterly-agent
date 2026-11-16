"""Command line: `python -m agent run` handles every open ticket once."""

import argparse
import sys

from agent.config import ConfigError, load_settings
from agent.faq import load_kb
from agent.helpdesk import FatalAPIError, HelpdeskClient, TransientExhausted
from agent.llm import LLMClient, LLMFatalError
from agent.runner import Runner


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m agent", description="Roosterly support ticket agent")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="decide and post every open ticket")
    run_parser.add_argument("--dry-run", action="store_true", help="decide and print, but post nothing")
    run_parser.add_argument("--limit", type=int, help="handle at most N tickets")
    commands.add_parser("status", help="show open and handled ticket counts")
    reset_parser = commands.add_parser("reset", help="re-open all practice tickets")
    reset_parser.add_argument("--yes", action="store_true", help="confirm: this wipes the previous run")
    args = parser.parse_args()

    try:
        settings = load_settings(require_llm=args.command == "run")
        helpdesk = HelpdeskClient(settings)
        if args.command == "status":
            print(helpdesk.status())
            return 0
        if args.command == "reset":
            if not args.yes:
                print("This re-opens all practice tickets and discards the previous run. Add --yes to confirm.")
                return 2
            print(helpdesk.reset_practice())
            return 0
        return run_agent(settings, helpdesk, args.dry_run, args.limit)
    except (ConfigError, FatalAPIError, LLMFatalError) as error:
        print(f"Cannot run: {error}", file=sys.stderr)
        return 2
    except TransientExhausted as error:
        print(
            f"The helpdesk is not responding ({error}). Nothing is lost; run the same command again.", file=sys.stderr
        )
        return 1


def run_agent(settings, helpdesk: HelpdeskClient, dry_run: bool, limit: int | None) -> int:
    summary = Runner(settings, helpdesk, LLMClient(settings), load_kb(), dry_run=dry_run).run(limit)
    print(f"\nRun {summary.run_id}: {summary.outcomes}")
    print(f"LLM cost this run: ${summary.cost_usd:.4f}")
    print(f"Helpdesk now: {summary.status.get('open_tickets')} open, {summary.status.get('handled_tickets')} handled")
    if summary.left_open:
        print(f"{summary.left_open} ticket(s) left open after a failure. Run the same command again.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
