# Scheduler design note: subsystem compatibility bounds

## Context

The scheduler layer is built from three independently replaceable subsystems:

| Subsystem | Abstract root | Directory |
| --- | --- | --- |
| Request storage | `RequestStorage[Payload]` | `storage/` |
| Client delivery | `ClientDelivery[Result]` | `delivery/` |
| Result retention | `ResultRetention[Result]` | `retention/` |

Each subsystem is a class hierarchy: an abstract root with increasingly
specialized subclasses beneath it. A concrete scheduler (`BaseScheduler`
subclass) is composed with one instance of each subsystem it needs.

Not every scheduler can work with every subsystem subclass. A scheduler relies
on specific guarantees — ordering, batching awareness, durability, cancellation
semantics — that only part of a hierarchy provides.

## Principle

**A scheduler declares, for each subsystem, the single highest-level (most
general) class it supports by using that class as the type of the corresponding
constructor parameter. This is a bounded type parameter — nothing more, nothing
hand-rolled. The type checker is the enforcement mechanism.**

`BaseScheduler` accepts each subsystem at its abstract root, so by default any
implementation is allowed:

```python
class BaseScheduler[Payload, Result](ABC):
    def __init__(
        self,
        storage: RequestStorage[Payload],
        delivery: ClientDelivery[Result],
    ) -> None: ...
```

A concrete scheduler that needs stronger guarantees **narrows the parameter type
to its supported root.** That annotation is the declaration:

```python
class DurableScheduler[Payload, Result](BaseScheduler[Payload, Result]):
    def __init__(
        self,
        storage: DurableRequestStorage[Payload],   # <- declared bound
        delivery: ClientDelivery[Result],          # <- still the abstract root
    ) -> None:
        super().__init__(storage, delivery)
```

The bound is an upper bound in the inheritance tree, and subtyping gives the rest
for free:

- The declared class is the *most general* type the parameter accepts.
- Any **descendant** (lower-level subclass) is accepted automatically, because it
  is a subtype — no extra code, the type checker allows it.
- Any **ancestor** is rejected, because it is not a subtype of the declared
  parameter type — the type checker flags it at the call site.

```
        RequestStorage            <- not a subtype of the bound, rejected
              |
     DurableRequestStorage        <- declared parameter type: accepted
          /         \
 RedisStorage   PostgresStorage   <- subtypes: accepted
```

Because the bound lives in the signature, construction with an unsupported
subsystem fails **statically**, at the call site, before anything runs. There is
no separate declaration to keep in sync and no runtime `isinstance` gate to
maintain.

## Multiple supported subtrees

A scheduler may support more than one disjoint subtree within the same
subsystem — it works with either of two unrelated families but not with their
common ancestor. Express this directly as a **union** of the subtree roots in the
parameter type:

```python
class FlexibleScheduler[Payload, Result](BaseScheduler[Payload, Result]):
    def __init__(
        self,
        storage: InMemoryStorageBase[Payload] | DurableRequestStorage[Payload],
        delivery: ClientDelivery[Result],
    ) -> None:
        super().__init__(storage, delivery)
```

Each member of the union carries the same rule: that root and its descendants are
accepted. The union accepts either subtree and everything beneath it, while still
rejecting the abstract `RequestStorage` root and any unlisted sibling subtree.

A single declared type remains the common case; a union is used only when the
supported set genuinely spans separate subtrees.

## Consequences and caveats

- **`super().__init__` stays typed at the root.** `BaseScheduler` still stores
  the subsystem at its abstract type; the subclass's narrower parameter type only
  constrains *what callers may pass in*. The base lifecycle code keeps working
  against the root interface.
- **Narrowing a parameter type in a subclass is deliberate.** A constructor is
  not part of the substitutable instance interface, so tightening its parameter
  type does not break Liskov substitution of scheduler *instances*. It does mean
  a subclass constructor accepts strictly fewer storage/delivery types than its
  parent — which is exactly the intended contract.
- **This covers linear hierarchies.** When a subsystem develops *orthogonal*
  guarantees (e.g. durable *and* ordered, independently), a single inheritance
  line can no longer express the bound; prefer a `Protocol` describing the
  required capabilities and annotate the parameter with that `Protocol` instead.
  The declaration mechanism is unchanged — it is still just the parameter type —
  only the type being named changes from a class to a capability protocol.

## Rationale

- **The contract and its enforcement are the same artifact.** The parameter type
  documents the weakest subsystem the scheduler depends on *and* enforces it, so
  the two cannot drift apart.
- **Substitutability is directional.** Subtyping already encodes "descendants in,
  ancestors out," so the type system does the work the design needs.
- **Unsupported compositions fail early and locally**, at the construction call
  site, rather than at runtime deep inside the scheduling loop.
