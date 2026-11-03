<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Python code style

## Overview

Python conventions for contributions to NVIDIA/xr-ai's SDK, utilities,
services, samples, and tests. These rules govern public interfaces, error
handling, and module layout. Application-owned code under `apps/` is outside
their scope.

## Scope and conventions

Package surfaces and their documented behavior live in
each package's `__all__` and docstrings; this reference covers only the
conventions that keep them consistent.

These rules describe the target state of all Python code. Code that does not yet
follow them is brought into line by cleanup changes and tracked outside this
reference. Making a supported public name conform to a rule is a breaking
change: keep the code working in every merged state, update every use in the
repository, and add a migration entry for released behavior. Until a scoped
change addresses it, the name keeps working as it is; an unrelated change is
not asked to rename it. Refer to {ref}`migration-release-rollover` for entry
scope.

A package that other projects import, such as everything under `agent-sdk/` and
`utils/` and any library a service or sample publishes, follows every section.
A runnable service, orchestrator, or sample worker publishes no API: keep its
modules private to the project and do not import them from elsewhere, except
from its own tests. The exception rules apply to all of them, and each entry
point is a sanctioned broad-catch site.

The rules apply established Python guidance, cited rule by rule. This reference
does not adopt any external style guide wholesale; for example, the repository's
worker packages use relative imports, which the Google guide discourages. Each
section ends with its **Basis**, which names the standard it comes from. A rule
marked **Repository convention** goes beyond a published standard to settle a
pattern that the repository handled inconsistently, and it has no external
source.

Review feedback on a change cites a rule from this reference. A preference
that is not written here is raised as a change to this reference, in its own
pull request, so that every contributor works from the same rules.

## Public and private code

The public surface of a package is the names in `__all__` of its top-level
`__init__.py`, plus the names in `__all__` of each module enrolled in the public
API inventory (`PUBLIC_API_MODULES` in `docs/source/_api_contract.py`). Imports
from an enrolled module, such as `xr_ai_tools.current_frame`,
`xr_ai_tools.vision`, `xr_ai_tools.rpc`, and `xr_ai_models.presets`, remain
supported. Nothing else is supported.

- Document every public name with a docstring, because the generated Python
  reference is rendered from it. Give a private name a docstring only when its
  invariant is not obvious.
- Start each private module with a short module docstring that states what
  it owns and which packages can import it.
- Use only public types in public signatures. Keep internal state in
  `_`-prefixed attributes that no other package reads. If a consumer needs
  that state, publish a deliberate accessor.
- Production code never imports a private name or module of another package.
  When a neighbouring package needs something, the owning package publishes it
  first. Implementation tests of the owning package can import private names.
- Keep transport and vendor details out of public names and documentation.
  Agent-facing text never mentions LiveKit or StreamKit.

**Basis:**
[PEP 8, Public and Internal Interfaces](https://peps.python.org/pep-0008/#public-and-internal-interfaces)
says "Documented interfaces are considered public" and that modules "should
explicitly declare the names in their public API using the `__all__`
attribute". It requires a leading underscore on internal interfaces and treats
an interface as internal when any containing namespace is internal. The public
API inventory and the contributing page's statement that public references are
generated from "each enrolled module's literal `__all__`" define which modules
are enrolled. The repository's
[`AGENTS.md`](https://github.com/NVIDIA/xr-ai/blob/main/AGENTS.md#architecture-boundaries)
requires that LiveKit not appear in agent APIs and that workers use public SDK
packages, and keeps private imports inside their owning packages or
implementation tests. The {doc}`contributing-and-conventions` change contract
places public API documentation in `__all__` and co-located docstrings. The
private-module docstring is a **repository convention**.

## Exports

Each public name has one canonical home, and these rules decide where it is.

- Every module is public or private. A public module is the package root or a
  module enrolled in `PUBLIC_API_MODULES`. Every other module has a leading
  underscore. Split a module that mixes public and internal code so the public
  names live in a public module.
- Export from the package root by default: define the name in a private module,
  import it explicitly in `__init__.py`, and list it in the root `__all__`.
- Add a public submodule only for one of two reasons, and state which in its
  module docstring. Either importing it requires an optional dependency declared
  as an extra, so the root could not import it without making the extra
  mandatory, as `xr_ai_tools.marker_tracking` does for OpenCV; or it namespaces
  a family of related names that would crowd the root, as `xr_ai_models.presets`
  does.
- A public module declares its names in a literal `__all__` and documents each
  one. The list holds names the module defines, and no underscore names. Only
  the package root re-exports names defined elsewhere. Do not use wildcard
  imports.
- Do not add a module to `PUBLIC_API_EXCLUSIONS`. A module with a public-looking
  name is either enrolled in the public API or given a leading underscore; the
  list only tracks modules awaiting that change. Check for importers before
  renaming one, and add a migration entry if a supported path moves.
- Import a name from its canonical home: the package root when the root exports
  it, otherwise the enrolled module that documents it.

**Basis:** The contract in `docs/source/_api_contract.py` already requires every
importable module to be enrolled or excluded and requires a literal `__all__`
in each public package and module. [PEP 8](https://peps.python.org/pep-0008/#imports)
says wildcard imports "should be avoided", because they make it unclear which
names are present. It also says that imported names "should always be
considered an implementation detail" unless they are an explicitly documented
part of the module's API, "such as ... a package's `__init__` module that
exposes functionality from submodules", which is why only the root re-exports.
The `xr-ai-tools` `pyproject.toml` declares the optional extras that justify
its public submodules. Exporting from the root by default, the two reasons for
a public submodule, and the rule against `PUBLIC_API_EXCLUSIONS` entries are
**repository conventions**.

## Module boundaries

The unit of encapsulation is a package, not a file. Hierarchy is fine: a package
can contain sub-packages, and each one is a boundary for the code around it.

- Import from a package, not from one of its files. Code outside a package
  imports from its root or from an enrolled public module, and never from a
  private file such as `xr_ai_voice._runtime`.
- A sub-package is a boundary for its parent. The parent imports from the
  sub-package's `__init__.py`, whose `__all__` lists what the parent may use,
  and not from the sub-package's files: write
  `from ._processors import VadSttProcessor`, not
  `from ._processors.vad_stt import VadSttProcessor`. Everything else in the
  sub-package is private to it.
- Modules in the same package form one unit and import each other directly with
  relative imports.
- Dependencies point one way. A sub-package does not import its parent's
  modules. Put a definition that both need in a lower-level module or
  sub-package that both import, or pass it in as an argument.
- An `__init__.py` assembles a boundary and holds no logic: a docstring,
  explicit imports, and `__all__`. Put logic in a private module and re-export
  what the boundary exposes.

**Basis:** [PEP 8](https://peps.python.org/pep-0008/#public-and-internal-interfaces)
treats an interface as internal when any containing namespace is internal, so
everything inside a private sub-package is internal, and it names "a package's
`__init__` module that exposes functionality from submodules" as the documented
way to expose names. PEP 8 also calls explicit relative imports "an acceptable
alternative to absolute imports, especially when dealing with complex package
layouts", and `AGENTS.md` requires relative internal imports in worker
packages. Treating the sub-package `__init__.py` as the parent's boundary, the
one-way dependency rule, and logic-free `__init__.py` files are **repository
conventions**.

## Exceptions

Every exception is either expected or unexpected, and the code makes the kind
obvious.

**Expected** failures are ones a caller can anticipate from the contract: a peer
is unreachable, a remote service returns an error status, external or
configuration input is malformed, a timeout fires, or a peer closes.

- Root a package's failures at a single package base class, such as
  `ModelServiceError`, with a few meaningful subclasses, and define each one
  next to the code that raises it. End every exception name in `Error`. Export
  the public ones through the package root or the enrolled module that raises
  them.
- List the SDK errors a public function raises in its docstring `Raises:`
  section. Do not list the `ValueError` or `TypeError` raised for a caller
  mistake, because that would make behavior under misuse part of the API.
- Raise where the failure is detected, and catch at the one place that can make
  a decision: retry, degrade, or report. Catch the narrowest type there, and do
  not catch in between.
- When a failure depends on state outside the program, such as a file, a socket,
  or an entry another task can change, attempt the operation and handle the
  specific exception instead of checking first, because the check can go stale
  before the operation runs. Validate caller-supplied arguments up front, and
  return a value for an ordinary outcome the caller branches on, so that an
  exception always marks a failure.
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
   `BaseException` outside the sanctioned sites below. A broad handler that only
   re-raises does nothing: delete it. Put work that must run whether or not the
   block fails in `finally`. Catch broadly and re-raise only to do work that
   applies to failure alone, such as rolling back partial state, and say why
   in a comment.
2. Do not silence an error with `pass`, `continue`, or a bare `return` in an
   `except` clause. Where ignoring a failure is correct, catch the specific
   type and explain why in a `# ignored:` comment.
3. The sanctioned broad-catch sites each carry a `# isolation boundary:`
   comment and consume the failure: the site logs it once with its traceback
   through the package's logger, then ends the unit of work or continues the
   loop it supervises. A handler that re-raises does not log; the owner that
   finally consumes the failure logs it. The sites are:
   - calling a user-supplied callback from a loop the SDK owns, so one handler
     cannot stop the loop;
   - a background-task supervisor or done callback that records the failure and
     surfaces it to the task's owner;
   - the process entry point.
4. Propagate a cancellation directed at the current task. Code that catches
   `asyncio.CancelledError` to clean up propagates it afterward, and prefers
   `try/finally` or `async with`. An owner that cancels a child task and awaits
   it can consume the child's `CancelledError` as the child's completion and
   continue its own cleanup or callback. Outside a block that is already
   propagating the owner's own cancellation, the owner first checks that
   `asyncio.current_task().cancelling()` is zero and otherwise re-raises,
   because the error can then come from the owner's own cancellation.
5. Log or raise at one layer, not both. A re-raised error is logged once, by
   the owner that consumes it.
6. Always preserve the cause with `raise ... from exc`. Make the message match
   the actual error condition, and say what the caller can do.
7. Cleanup that must not mask the primary error uses
   `contextlib.suppress(<specific type>)` with a comment.
8. Keep each `try` body to the statements that can raise the handled error.
9. Release resources with `with` or `async with`, using `contextlib.closing` for
   an object that lacks one. Use `try/finally` for other cleanup.

**Basis:**
- Rules 1 and 2, and the sanctioned sites in rule 3, follow the
  [Google Python Style Guide, section 2.4.4](https://google.github.io/styleguide/pyguide.html#s2.4.4-decision),
  which forbids catch-all `except:` and catching `Exception` unless re-raising
  or "creating an isolation point in the program where exceptions are not
  propagated but are recorded and suppressed instead, such as protecting a
  thread from crashing by guarding its outermost block". Rule 1 narrows the
  re-raising allowance to handlers that do failure-only work, because a broad
  handler that merely re-raises has no effect; that narrowing is a
  **repository convention**. The same section says a library's own exceptions
  "must inherit from an existing exception class" and that names end in `Error`;
  calls for `finally` for cleanup; advises raising `ValueError` for a violated
  precondition; and says not to use `assert` to validate preconditions. Its
  [function docstring rules](https://google.github.io/styleguide/pyguide.html#s3.8.3-functions-and-methods)
  say not to document exceptions raised when the API is violated, and its
  [error message rules](https://google.github.io/styleguide/pyguide.html#s3.10.2-error-messages)
  require that a message "precisely match the actual error condition". The
  [PEP 8 programming recommendations](https://peps.python.org/pep-0008/#programming-recommendations)
  warn that a bare `except:` "can disguise other problems", direct the author to
  name specific exceptions, limit each `try` clause to the minimum code (rule
  8), and use `raise X from Y` to replace an exception without losing the
  traceback (rule 6). [PEP 20](https://peps.python.org/pep-0020/) states the
  principle: "Errors should never pass silently. Unless explicitly silenced."
- Ruff has matching rules:
  [`BLE001`](https://docs.astral.sh/ruff/rules/blind-except/) (blind except),
  [`S110`](https://docs.astral.sh/ruff/rules/try-except-pass/) (try-except-pass),
  and [`B904`](https://docs.astral.sh/ruff/rules/raise-without-from-inside-except/)
  (missing `from`).
- Rule 4 follows the asyncio documentation on
  [task cancellation](https://docs.python.org/3/library/asyncio-task.html#task-cancellation):
  it recommends `try/finally` for cleanup, says a caught `CancelledError`
  "should generally be propagated when clean-up is complete", and warns that
  `asyncio.TaskGroup` and `asyncio.timeout()` "might misbehave if a coroutine
  swallows" it. The documentation defines
  [`Task.cancelling()`](https://docs.python.org/3/library/asyncio-task.html#asyncio.Task.cancelling)
  as "the number of pending cancellation requests to this Task", which is what
  the owner check reads. Letting an owner consume a child's completion is a
  **repository convention**; the existing shutdown paths in `xr_ai_voice` and
  `xr_ai_tools` cancel and await child tasks this way.
- The attempt-first bullet follows the Python glossary definition of
  [EAFP](https://docs.python.org/3/glossary.html#term-EAFP), a style that
  "assumes the existence of valid keys or attributes and catches exceptions if
  the assumption proves false". The
  [`os.access` documentation](https://docs.python.org/3/library/os.html#os.access)
  says that checking before opening "creates a security hole, because the user
  might exploit the short time interval between checking and opening the file"
  and that "It's preferable to use EAFP techniques". Validating arguments up
  front is the Google guide's advice to raise `ValueError` for a violated
  precondition.
- Rule 9 follows the Google guide's
  [section 3.11](https://google.github.io/styleguide/pyguide.html#s3.11-files-sockets-closeables):
  "The preferred way to manage files and similar resources is using the `with`
  statement", with `contextlib.closing()` for objects that do not support it.
- Rule 7 follows the
  [`contextlib.suppress` documentation](https://docs.python.org/3/library/contextlib.html#contextlib.suppress),
  which limits it to "very specific errors where silently continuing with
  program execution is known to be the right thing to do".
- **Repository conventions:** raising where the failure is detected and catching
  only where a decision can be made, returning a value for an ordinary outcome
  so exceptions mark failures only, the per-package base class (the Google guide
  requires only that an exception inherit from an existing one), inheriting the
  replaced builtin to avoid a breaking change, translating foreign errors
  exactly once, the log-once rule (rule 5), and the greppable
  `# isolation boundary:` and `# ignored:` comment tags.

## API design

- Follow {ref}`resource-lifecycle` for ownership, thread cancellation, and
  shutdown budgets.
- Do not model a behavior with a boolean flag plus an optional argument. Use one
  typed choice. Use an `Enum` or `Literal` instead of a string for a kind or
  state.
- Make lifecycle explicit and symmetric: `async with`, or `start()` and
  `stop()`. `stop()` is idempotent, wakes the loop it stops, and survives
  repeated cancellation. A constructor does not perform I/O or start tasks.
- Return an unsubscribe callable from every `on_*` registration. State in the
  docstring which loop runs the callback and what happens if it raises.
- Do not return `None` to mean a failed operation; a lookup that can find
  nothing can return `None`. Raise a typed error, or offer an
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
prefers the `with` statement for resources and requires that code documentation
"explain clearly how resource lifetime is managed" when that is infeasible,
which is the reason for explicit lifecycle. The flag rule applies Martin
Fowler's [Flag Argument](https://martinfowler.com/bliki/FlagArgument.html),
which prefers separate, explicit calls because they "communicate more clearly
what my intention is". The second-way rule restates the repository's own
`AGENTS.md`
[engineering guidance](https://github.com/NVIDIA/xr-ai/blob/main/AGENTS.md#engineering-guidance):
"Do not add an abstraction until two concrete use cases need it." The callback,
`try_*`, idempotent `stop()`, and constructor-policy rules, and the preference
for `Enum` or `Literal`, are **repository conventions**.

## Comments

Comments explain a non-obvious invariant or failure mode. Use the fixed tags
`# isolation boundary:` and `# ignored:` so policy sites are easy to find with
a search.

**Basis:**
[`AGENTS.md`, Comments and documentation](https://github.com/NVIDIA/xr-ai/blob/main/AGENTS.md#comments-and-documentation)
limits comments to non-obvious invariants and failure modes. The tags are a
**repository convention**.

## Structuring a change

- Follow the upstream {ref}`pull-request-scope` rules.
- Do not mix a file move with an edit. Put the move in its own commit so
  `git diff -M` shows it as a pure rename.
- When splitting a module, keep the old import path working through a
  re-export until every in-repository importer, including tests, has moved.
  Remove the re-export in a follow-up change. Removing an import path that is
  supported, such as an enrolled module, is a breaking change.
- State in the pull request whether the change alters behavior. A refactor that
  must not change behavior passes the existing tests unmodified.
- Add a migration entry for a breaking change to released behavior.

**Basis:** Google's engineering practices on
[small changes](https://google.github.io/eng-practices/review/developer/small-cls.html)
say the right size is "one self-contained change" and that refactorings
belong in a separate change from bug fixes, with "moving and renaming a class"
as the example. The migration entry is required by the contributing guide's
change contract; refer to {ref}`migration-release-rollover` for baseline
verification and entry scope. The re-export step is a **repository convention**.
