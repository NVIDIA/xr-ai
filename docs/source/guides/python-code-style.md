<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Python code style

This reference governs Python code throughout the repository: where the public
API ends, how failures are represented, and how modules are laid out. It covers
the SDK, utilities, services, samples, and tests. Application-owned projects
under `apps/` are outside repository tooling and outside this page. Package
surfaces and their documented behavior live in each package's `__all__` and
docstrings; this page covers only the conventions that keep them consistent.

A package that other projects import, such as everything under `agent-sdk/` and
`utils/` and any library a service or sample publishes, follows every section.
A runnable service, orchestrator, or sample worker publishes no API: keep its
modules private to the project and do not import them from elsewhere. The
exception rules apply to all of them, and its entry point is a sanctioned
broad-catch site.

The rules apply established Python guidance, cited rule by rule. This page does
not adopt any external style guide wholesale; for example, the repository's
worker packages use relative imports, which the Google guide discourages. Each section ends with its
**Basis**, which names the standard it comes from. A rule marked **Repository
convention** goes beyond a published standard to settle a pattern that the
SDK handled inconsistently, and it has no external source.

Review feedback on an SDK change cites a rule from this page. A preference that
is not written here is raised as a change to this page, in its own pull
request, so that every contributor works from the same rules.

## Public and private code

The public surface of a package is exactly the names in `__all__` of its
top-level `__init__.py`. Nothing else is supported.

- Name every module that is not re-exported with a leading underscore. A module
  that mixes public and internal code is split so the public names are the only
  ones a consumer needs to import.
- Document every public name with a docstring, because the generated Python
  reference is rendered from it. Give a private name a docstring only when its
  invariant is not obvious.
- Start each private module with a short module docstring that states what it
  owns and which packages can import it.
- Use only public types in public signatures. Keep internal state in
  `_`-prefixed attributes that no other package reads. If a consumer needs
  that state, publish a deliberate accessor.
- Never import a private name or module of another package. When a neighbouring
  package needs something, the owning package publishes it first.
- Import public names from the package root. Treat a name that consumers need
  from a submodule as a missing `__all__` entry.
- Keep transport and vendor details out of public names and documentation.
  Agent-facing text never mentions LiveKit or StreamKit.

**Basis:**
[PEP 8, Public and Internal Interfaces](https://peps.python.org/pep-0008/#public-and-internal-interfaces)
defines the public surface as the names declared in `__all__`, requires a
leading underscore on internal interfaces, and treats an interface as internal
when any containing namespace is internal. The repository's
[`AGENTS.md`](https://github.com/NVIDIA/xr-ai/blob/main/AGENTS.md#architecture-boundaries)
requires that LiveKit not appear in agent APIs and that workers use public SDK
packages, and its change contract places public API documentation in
`__all__` and co-located docstrings. Importing from the package root and the
private-module docstring are **repository conventions**.

## Exceptions

Every exception is either expected or unexpected, and the code makes the kind
obvious.

**Expected** failures are ones a caller can anticipate from the contract: a peer
is unreachable, a remote service returns an error status, external or
configuration input is malformed, a timeout fires, or a peer closes.

- Define a package's failures in one `_errors.py` module rooted at a single
  package base class, such as `ModelServiceError`, with a few meaningful
  subclasses. End every exception name in `Error`. Re-export the public ones in
  `__all__`.
- List the SDK errors a public function raises in its docstring `Raises:`
  section. Do not list the `ValueError` or `TypeError` raised for a caller
  mistake, because that would make behavior under misuse part of the API.
- Catch the narrowest type, at the one place that can make a decision: retry,
  degrade, or report.
- Translate a foreign error (`httpx`, `zmq`, `grpc`, `OSError`, `json`, `yaml`,
  `pydantic`) into an SDK error exactly once, at the adapter boundary, with
  `raise SdkError(...) from exc`. Foreign types do not appear in public
  signatures.
- Raise `ValueError` or `TypeError` eagerly for caller mistakes. These are
  programming errors, not expected failures. Do not use `assert` to validate
  arguments or object state; an `assert` can be removed without changing the
  program, so nothing may depend on it.
- When replacing a builtin that callers already catch, make the new class also
  inherit that builtin so the change is not breaking.

**Unexpected** failures are bugs or violated invariants: a `KeyError` on an
internal mapping, an `AttributeError`, an `AssertionError`. Do not catch them
to hide them. They propagate to the process top level, which logs once with a
traceback and exits non-zero. A visible crash is easier to diagnose than a
swallowed error.

These rules follow:

1. Do not write a bare `except:`, and do not catch `Exception` or
   `BaseException` outside the sanctioned sites below.
2. Do not silence an error with `pass`, `continue`, or a bare `return` in an
   `except` clause. Where ignoring a failure is correct, catch the specific
   type and explain why in a `# ignored:` comment.
3. The sanctioned broad-catch sites each carry a `# isolation boundary:`
   comment, log with `logger.exception`, and either re-raise or end the unit of
   work they supervise:
   - calling a user-supplied callback from a loop the SDK owns, so one handler
     cannot stop the loop;
   - a background-task supervisor or done callback that records the failure and
     surfaces it to the task's owner;
   - the process entry point.
4. Do not suppress `asyncio.CancelledError`. When code catches it to clean up,
   it propagates the error afterward. Prefer `try/finally` or `async with` for
   cleanup.
5. Log or raise at one layer, not both. A re-raised error is logged once, at
   the top.
6. Always preserve the cause with `raise ... from exc`. Make the message match
   the actual error condition, and say what the caller can do.
7. Cleanup that must not mask the primary error uses
   `contextlib.suppress(<specific type>)` with a comment.
8. Keep each `try` body to the statements that can raise the handled error.

**Basis:**
- Rules 1 and 2, and the sanctioned sites in rule 3, follow the
  [Google Python Style Guide, section 2.4.4](https://google.github.io/styleguide/pyguide.html#s2.4.4-decision),
  which forbids catch-all `except:` and catching `Exception` unless re-raising or
  "creating an isolation point in the program where exceptions are not
  propagated but are recorded and suppressed instead, such as protecting a
  thread from crashing by guarding its outermost block". The same section
  says a library's own exceptions "must inherit from an existing exception
  class" and that names should end in `Error`; calls for `finally` for cleanup;
  advises raising `ValueError` for a violated precondition; and says not to use
  `assert` to validate preconditions. Its
  [function docstring rules](https://google.github.io/styleguide/pyguide.html#s3.8.3-functions-and-methods)
  say not to document exceptions raised when the API is violated, and its
  [error message rules](https://google.github.io/styleguide/pyguide.html#s3.10.2-error-messages)
  require that a message "precisely match the actual error condition". The
  [PEP 8 programming recommendations](https://peps.python.org/pep-0008/#programming-recommendations)
  warn that a bare `except:` "can disguise other problems", direct the author to
  name specific exceptions, limit each `try` clause to the minimum code
  (rule 8), and use `raise X from Y` to replace an exception without losing the
  traceback (rule 6). [PEP 20](https://peps.python.org/pep-0020/) states the
  principle: "Errors should never pass silently. Unless explicitly silenced."
- Ruff enforces the same rules mechanically:
  [`BLE001`](https://docs.astral.sh/ruff/rules/blind-except/) (blind except),
  [`S110`](https://docs.astral.sh/ruff/rules/try-except-pass/) (try-except-pass),
  and [`B904`](https://docs.astral.sh/ruff/rules/raise-without-from-inside-except/)
  (missing `from`).
- Rule 4 follows the asyncio documentation on
  [task cancellation](https://docs.python.org/3/library/asyncio-task.html#task-cancellation):
  it recommends `try/finally` for cleanup, says a caught `CancelledError`
  "should generally be propagated when clean-up is complete", and warns that
  `asyncio.TaskGroup` and `asyncio.timeout()` "might misbehave if a coroutine
  swallows" it.
- Rule 7 follows the
  [`contextlib.suppress` documentation](https://docs.python.org/3/library/contextlib.html#contextlib.suppress),
  which limits it to "very specific errors where silently continuing with
  program execution is known to be the right thing to do".
- **Repository conventions:** the per-package base class in `_errors.py` (the
  Google guide requires only that an exception inherit from an existing one),
  inheriting
  the replaced builtin to avoid a breaking change, translating foreign errors
  exactly once, the log-once rule (rule 5), and the greppable
  `# isolation boundary:` and `# ignored:` comment tags.

## API design

- Do not model a behavior with a boolean flag plus an optional argument. Use one
  typed choice. Use an `Enum` or `Literal` instead of a string for a kind or
  state.
- Make lifecycle explicit and symmetric: `async with`, or `start()` and
  `stop()`. `stop()` is idempotent, wakes the loop it stops, and survives
  repeated cancellation. A constructor does not perform I/O or start tasks.
- Return an unsubscribe callable from every `on_*` registration. State in the
  docstring which loop runs the callback and what happens if it raises.
- Do not return `None` to mean failure. Raise a typed error, or offer an
  explicit `try_*` variant that returns a `bool`.
- Expose attributes read-only unless mutation is the documented purpose.
- Do not add a second way to do something, such as a synchronous twin or a
  parallel constructor, until two in-tree callers need it.
- Pass policy to a constructor. Do not assign private attributes on another
  object after constructing it.

**Basis:** The Google guide supports two of the rules. Section
[2.13](https://google.github.io/styleguide/pyguide.html#s2.13-properties) lists
making an attribute read-only as a use of `@property`, and section
[3.11](https://google.github.io/styleguide/pyguide.html#s3.11-files-sockets-closeables)
prefers the `with` statement for resources and requires that code
documentation "explain clearly how resource lifetime is managed" when that is
infeasible, which is the reason for explicit lifecycle. The flag rule applies Martin Fowler's
[Flag Argument](https://martinfowler.com/bliki/FlagArgument.html), which
prefers separate, explicit calls because they "communicate more clearly what
my intention is". The second-way rule restates the repository's own
`AGENTS.md` [change contract](https://github.com/NVIDIA/xr-ai/blob/main/AGENTS.md#change-contract):
"Do not add an abstraction until two concrete use cases need it." The
callback, `try_*`, idempotent `stop()`, and constructor-policy rules, and the
preference for `Enum` or `Literal`, are **repository conventions**.

## Comments

Comments explain a non-obvious invariant or failure mode. Use the fixed tags
`# isolation boundary:` and `# ignored:` so policy sites are easy to find with
a search.

**Basis:**
[`AGENTS.md`, Comments and documentation](https://github.com/NVIDIA/xr-ai/blob/main/AGENTS.md#comments-and-documentation)
limits comments to non-obvious invariants and failure modes. The tags are a
**repository convention**.

## Structuring a change

- Keep one concern per pull request, ideally under 300 changed lines.
- Do not mix a file move with an edit. Put the move in its own commit so
  `git diff -M` shows it as a pure rename.
- When splitting a module, keep the old import path working through a
  re-export until every in-repository importer, including tests, has moved.
  Remove the re-export in a follow-up change.
- State in the pull request whether the change alters behavior. A refactor that
  must not change behavior passes the existing tests unmodified.
- Add a migration entry for a breaking change.

**Basis:** Google's engineering practices on
[small changes](https://google.github.io/eng-practices/review/developer/small-cls.html)
say the right size is "one self-contained change" and that refactorings
belong in a separate change from bug fixes, with "moving and renaming a class"
as the example. The migration entry is required by the `AGENTS.md` change
contract. The 300-line guideline and the re-export step are **repository
conventions**.
