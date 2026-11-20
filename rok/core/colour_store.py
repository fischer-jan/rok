"""The scene's colours: entries with stable ids, your order, and the slots
voxels store.

A voxel holds a slot number (1 to 255). A colour is an entry: an id that
never changes, an RGBA colour, a material, a name and a group. An entry holds
a slot (a lease) while voxels use it or while it is the colour you paint with;
an unused entry holds none, so no other colour can overwrite it. `order` is
the strip, every entry in the order you gave them.

The 256-row palette and the material table are what the renderer, the
shaders and the .vox writer read. Only the store writes them (`publish`), in
place, because every grid shares the palette array.

A command asks for colours with `plan`, which finds each colour by value or
makes a new entry and gives it a free slot, and then `commit`s them in one
go. A free slot has no voxel in any model or frame and no lease that must
stay (the colour you paint with). Leases of slots without voxels are taken
back when a plan needs the room; their entries keep their colour.

The width lives in `SLOTS`. numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .material import DEFAULT, Material, MaterialTable
from .palette import DEFAULT_PALETTE

SLOTS = 256  # slot 0 is empty; a voxel holds 0..SLOTS-1

RGBA = tuple[int, int, int, int]


def rgba(colour) -> RGBA:
    """A colour as four ints; three values are opaque."""
    values = [int(c) for c in colour]
    if len(values) == 3:
        values.append(255)
    return (values[0], values[1], values[2], values[3])


@dataclass(frozen=True)
class Entry:
    rgba: RGBA
    material: Material = DEFAULT
    name: str = ""
    group: str = ""

    @property
    def key(self) -> tuple[RGBA, Material]:
        """What makes two entries the same colour."""
        return (self.rgba, self.material)


@dataclass
class Plan:
    """What `plan` found for a list of wanted colours, before anything changes.

    `ids[k]` is wanted colour k's entry: an existing id, or a new one listed
    in `new`. `leases` are the slots new or unused entries get; `reclaim` the
    entries whose slot is taken back for them. `short` counts the colours that
    found no slot: the caller asks to merge them (`merged`) or cancels."""

    ids: list[int] = field(default_factory=list)
    new: dict[int, Entry] = field(default_factory=dict)
    leases: dict[int, int] = field(default_factory=dict)  # slot -> id
    reclaim: list[int] = field(default_factory=list)
    short: int = 0
    merged: dict[int, int] = field(default_factory=dict)  # k -> the id it merged into

    def slot_map(self, store: ColourStore) -> list[int]:
        """The slot each wanted colour ends on once the plan is committed."""
        out = []
        lease_of = {i: s for s, i in self.leases.items()}
        for i in self.ids:
            out.append(lease_of.get(i, store.slot_of(i) or 0))
        return out


@dataclass
class Snapshot:
    """The store at one moment, for undo."""

    entries: dict[int, Entry]
    order: list[int]
    id_at: list[int | None]
    next_id: int
    palette: np.ndarray | None = None  # every row, also those without a lease
    materials: MaterialTable | None = None


class ColourStore:
    def __init__(self, palette: np.ndarray, materials: MaterialTable) -> None:
        self.palette = palette  # the projection: shared, written in place
        self.materials = materials
        self.entries: dict[int, Entry] = {}
        self.order: list[int] = []
        self.id_at: list[int | None] = [None] * SLOTS
        self.next_id = 1
        self.version = 0  # bumped on every change; the undo hook compares it
        self._seen: tuple[np.ndarray, MaterialTable] | None = None  # the palette as the store last left it
        self.on_sync: list = []  # f(version before): told after `sync` took on outside changes

    # --- building -----------------------------------------------------------

    @classmethod
    def from_palette(cls, palette: np.ndarray, materials: MaterialTable, used: np.ndarray,
                     keep_unused: bool = True) -> ColourStore:
        """A store for a palette that has none yet (a .vox): each
        used slot becomes an entry on its slot, in slot order; then, with
        `keep_unused`, each unused slot that differs from the default palette
        becomes an unused entry. Equal colours on several used slots stay
        separate entries, so no voxel changes."""
        store = cls(palette, materials)
        used = np.asarray(used)
        seen: set[tuple[RGBA, Material]] = set()
        for slot in range(1, SLOTS):
            if int(used[slot]) > 0:
                entry = Entry(rgba(palette[slot]), materials.get(slot))
                ident = store._add(entry)
                store.id_at[slot] = ident
                seen.add(entry.key)
        if keep_unused:
            for slot in range(1, SLOTS):
                if int(used[slot]) > 0:
                    continue
                entry = Entry(rgba(palette[slot]), materials.get(slot))
                untouched = entry.material == DEFAULT and np.array_equal(palette[slot], DEFAULT_PALETTE[slot])
                if untouched or entry.rgba[3] == 0 or entry.key in seen:
                    continue
                store._add(entry)
                seen.add(entry.key)
        return store

    def _add(self, entry: Entry, after: int | None = None) -> int:
        ident = self.next_id
        self.next_id += 1
        self.entries[ident] = entry
        if after is not None and after in self.order:
            self.order.insert(self.order.index(after) + 1, ident)
        else:
            self.order.append(ident)
        return ident

    # --- queries ------------------------------------------------------------

    def slot_of(self, ident: int) -> int | None:
        for slot in range(1, SLOTS):
            if self.id_at[slot] == ident:
                return slot
        return None

    def id_of(self, slot: int) -> int | None:
        return self.id_at[int(slot)] if 0 < int(slot) < SLOTS else None

    def find(self, colour, material: Material | None = None) -> int | None:
        """The entry with exactly this colour and material; a leased one first."""
        key = (rgba(colour), DEFAULT if material is None else material)
        found = None
        for ident in self.order:
            if self.entries[ident].key == key:
                if self.slot_of(ident) is not None:
                    return ident
                if found is None:
                    found = ident
        return found

    def used_ids(self, used: np.ndarray) -> list[int]:
        """The entries voxels use, in strip order."""
        used = np.asarray(used)
        on = {self.id_at[s] for s in range(1, SLOTS) if int(used[s]) > 0}
        return [i for i in self.order if i in on]

    def unused_ids(self, used: np.ndarray) -> list[int]:
        on = set(self.used_ids(used))
        return [i for i in self.order if i not in on]

    # --- planning -----------------------------------------------------------

    def free_slots(self, used: np.ndarray, pinned=()) -> list[int]:
        """Slots a new lease may take: no voxel, and no lease, or a lease of
        an entry that is not pinned (it is taken back). Slots without any
        lease come first, low to high."""
        used = np.asarray(used)
        pins = {int(p) for p in pinned}
        empty, reclaimable = [], []
        for slot in range(1, SLOTS):
            if int(used[slot]) > 0 or slot in pins:
                continue
            (empty if self.id_at[slot] is None else reclaimable).append(slot)
        return empty + reclaimable

    def plan(self, wants, used: np.ndarray, pinned=()) -> Plan:
        """Entries and slots for `wants`, a list of colours or (colour,
        material) pairs. A colour equal to an entry reuses it (and its slot,
        or a new lease when it has none); others become new entries. Two
        wants of one colour get one entry. Changes nothing."""
        out = Plan()
        free = self.free_slots(used, pinned)
        pending: dict[tuple[RGBA, Material], int] = {}
        for want in wants:
            colour, material = want if _is_pair(want) else (want, None)
            entry = Entry(rgba(colour), DEFAULT if material is None else material)
            ident = pending.get(entry.key) or self.find(entry.rgba, entry.material)
            if ident is None:
                ident = -(len(out.new) + 1)  # a new entry; numbered on commit
                out.new[ident] = entry
            pending[entry.key] = ident
            out.ids.append(ident)
            if self.slot_of(ident) is not None or ident in out.leases.values():
                continue
            slot = next((s for s in free if s not in out.leases), None)
            if slot is None:
                out.short += 1
                continue
            out.leases[slot] = ident
            if self.id_at[slot] is not None:
                out.reclaim.append(self.id_at[slot])
        return out

    def merge_short(self, plan: Plan, used: np.ndarray) -> None:
        """Send every wanted colour that found no slot to the nearest colour
        voxels use (OKLab), of the same material kind when there is one."""
        leased = set(plan.leases.values()) | set(self.used_ids(used))
        for k, ident in enumerate(plan.ids):
            if self.slot_of(ident) is not None or ident in plan.leases.values():
                continue
            entry = plan.new.get(ident) or self.entries[ident]
            options = [i for i in leased if i > 0]
            same = [i for i in options if self.entries[i].material.type == entry.material.type]
            target = nearest(entry.rgba, [(i, self.entries[i].rgba) for i in (same or options)])
            if target is not None:
                plan.merged[k] = target
                plan.ids[k] = target
        plan.short = 0
        plan.new = {i: e for i, e in plan.new.items() if i in plan.ids}

    def commit(self, plan: Plan, after: int | None = None) -> list[int]:
        """Make the plan real. New entries go into the strip after `after`
        (the colour they came from), in order; reclaimed entries lose their
        slot. Returns the final id of each wanted colour."""
        if plan.short:
            raise ValueError("plan has colours without a slot; merge or cancel first")
        final: dict[int, int] = {}
        anchor = after
        for temp, entry in plan.new.items():
            final[temp] = self._add(entry, anchor)
            anchor = final[temp]
        for slot, ident in plan.leases.items():
            self.id_at[slot] = final.get(ident, ident)
            self.publish(slot)
        self.version += 1
        return [final.get(i, i) for i in plan.ids]

    def lease(self, ident: int, used: np.ndarray, pinned=()) -> int | None:
        """A slot for entry `ident` (the colour you paint with): its own, or a
        free one. None when every slot holds voxels."""
        slot = self.slot_of(ident)
        if slot is not None:
            return slot
        free = self.free_slots(used, pinned)
        if not free:
            return None
        slot = free[0]
        self.id_at[slot] = ident
        self.publish(slot)
        self.version += 1
        return slot

    def add(self, colour, material: Material | None = None, after: int | None = None, name: str = "") -> int:
        """A new unused colour (no slot), after `after` in the strip or at its end."""
        ident = self._add(Entry(rgba(colour), DEFAULT if material is None else material, name), after)
        self.version += 1
        return ident

    def claim(self, slot: int, colour, material: Material | None = None, after: int | None = None) -> int:
        """Put a colour on a slot chosen elsewhere (a free one: no voxel holds
        it). An entry of exactly this colour without a slot takes it; else a
        new entry, after `after` in the strip. The slot's old entry loses it."""
        entry = Entry(rgba(colour), DEFAULT if material is None else material)
        ident = self.find(entry.rgba, entry.material)
        if ident is None or self.slot_of(ident) is not None:
            ident = self._add(entry, after)
        self.id_at[int(slot)] = ident
        self.publish(int(slot))
        self.version += 1
        return ident

    # --- editing ------------------------------------------------------------

    def edit(self, ident: int, colour=None, material: Material | None = None, name: str | None = None) -> None:
        """Change an entry: every voxel of it changes with it. The one way to
        recolour voxels."""
        entry = self.entries[ident]
        if colour is not None:
            entry = replace(entry, rgba=rgba(colour))
        if material is not None:
            entry = replace(entry, material=material)
        if name is not None:
            entry = replace(entry, name=name)
        self.entries[ident] = entry
        slot = self.slot_of(ident)
        if slot is not None:
            self.publish(slot)
        self.version += 1

    def move(self, ident: int, index: int) -> None:
        """Put entry `ident` at strip position `index`. No voxel changes."""
        self.order.remove(ident)
        self.order.insert(max(0, min(int(index), len(self.order))), ident)
        self.version += 1

    def remove(self, ids, used: np.ndarray) -> list[int]:
        """Delete unused entries; an entry voxels use stays. Returns the ids removed."""
        on = set(self.used_ids(used))
        gone = [i for i in ids if i in self.entries and i not in on]
        for ident in gone:
            slot = self.slot_of(ident)
            if slot is not None:
                self.id_at[slot] = None
            del self.entries[ident]
            self.order.remove(ident)
        if gone:
            self.version += 1
        return gone

    def _mark(self) -> None:
        self._seen = (self.palette.copy(), self.materials.copy())

    def sync(self) -> bool:
        """Take on palette rows that changed outside the store (an old path,
        a script): a slot with a colour gets the new colour and material on
        its entry; a slot without one becomes a colour on that slot. True
        when something changed."""
        if self._seen is None:
            self._mark()
            return False
        rows, table = self._seen
        if np.array_equal(self.palette, rows) and self.materials == table:
            return False
        changed = False
        before = self.version
        for slot in range(1, SLOTS):
            row = rgba(self.palette[slot])
            material = self.materials.get(slot)
            if row == rgba(rows[slot]) and material == table.get(slot):
                continue
            ident = self.id_at[slot]
            if ident is not None:
                entry = self.entries[ident]
                if entry.rgba != row or entry.material != material:
                    self.entries[ident] = replace(entry, rgba=row, material=material)
                    changed = True
            elif row[3] > 0:
                self.id_at[slot] = self._add(Entry(row, material))
                changed = True
        if changed:
            self.version += 1
        self._mark()
        for f in self.on_sync:
            f(before)
        return changed

    def adopt_used(self, used: np.ndarray) -> bool:
        """Every slot voxels use gets a colour: a slot without one (painted
        before the store knew it) becomes a colour with its palette row."""
        used = np.asarray(used)
        missing = [s for s in range(1, SLOTS) if int(used[s]) > 0 and self.id_at[s] is None]
        for slot in missing:
            self.id_at[slot] = self._add(Entry(rgba(self.palette[slot]), self.materials.get(slot)))
        if missing:
            self.version += 1
        return bool(missing)

    def compact(self, used: np.ndarray) -> tuple[np.ndarray, np.ndarray, MaterialTable]:
        """The colours voxels use, in your order, on slots 1, 2, ... for a
        .vox file: (lut from your slots to the file's, the file's palette,
        its material table). The rest of the file's palette is the default
        palette, so it reads as empty again. The store does not change."""
        lut = np.zeros(SLOTS, dtype=np.uint8)
        palette = np.array(DEFAULT_PALETTE, copy=True)
        table = MaterialTable()
        for new, ident in enumerate(self.used_ids(used), start=1):
            old = self.slot_of(ident)
            entry = self.entries[ident]
            lut[old] = new
            palette[new] = entry.rgba
            if entry.material != DEFAULT:
                table.set(new, entry.material)
        return lut, palette, table

    def blank(self, slots) -> None:
        """Empty slots no voxel uses: transparent black, plain, no entry."""
        for slot in slots:
            self.id_at[int(slot)] = None
            self.palette[int(slot)] = (0, 0, 0, 0)
            if self.materials.get(int(slot)) != DEFAULT:
                self.materials.set(int(slot), DEFAULT)
        self._mark()
        self.version += 1

    def publish(self, slot: int) -> None:
        """Write slot `slot`'s entry into the palette and the material table."""
        ident = self.id_at[slot]
        if ident is None:
            return
        entry = self.entries[ident]
        self.palette[slot] = entry.rgba
        if self.materials.get(slot) != entry.material:
            self.materials.set(slot, entry.material)
        if self._seen is not None:
            self._seen[0][slot] = self.palette[slot]
            if self._seen[1].get(slot) != entry.material:
                self._seen[1].set(slot, entry.material)

    # --- undo ---------------------------------------------------------------

    def snapshot(self) -> Snapshot:
        return Snapshot(dict(self.entries), list(self.order), list(self.id_at), self.next_id,
                        self.palette.copy(), self.materials.copy())

    def restore(self, snap: Snapshot) -> None:
        self.entries = dict(snap.entries)
        self.order = list(snap.order)
        self.id_at = list(snap.id_at)
        self.next_id = snap.next_id
        if snap.palette is not None:
            self.palette[...] = snap.palette  # in place: every grid shares the array
            self.materials.restore(snap.materials)
        else:
            for slot in range(1, SLOTS):
                self.publish(slot)
        self._mark()
        self.version += 1

    # --- files --------------------------------------------------------------

    def to_json(self) -> dict:
        return {
            "entries": [[i, list(e.rgba), _material_json(e.material), e.name, e.group]
                        for i, e in ((i, self.entries[i]) for i in self.order)],
            "slots": {str(s): i for s, i in enumerate(self.id_at) if i is not None},
            "next_id": self.next_id,
        }

    @classmethod
    def from_json(cls, data: dict, palette: np.ndarray, materials: MaterialTable) -> ColourStore:
        store = cls(palette, materials)
        for ident, colour, material, name, group in data["entries"]:
            store.entries[int(ident)] = Entry(rgba(colour), _material_from(material), str(name), str(group))
            store.order.append(int(ident))
        for slot, ident in data.get("slots", {}).items():
            if int(ident) in store.entries and 0 < int(slot) < SLOTS:
                store.id_at[int(slot)] = int(ident)
        store.next_id = max([int(data.get("next_id", 1)), *(i + 1 for i in store.entries)])
        for slot in range(1, SLOTS):
            store.publish(slot)
        return store

    def copy(self) -> ColourStore:
        """A scratch store on copies of the palette and table, for previews."""
        out = ColourStore(self.palette.copy(), self.materials.copy())
        out.entries = dict(self.entries)
        out.order = list(self.order)
        out.id_at = list(self.id_at)
        out.next_id = self.next_id
        return out


def _is_pair(want) -> bool:
    return isinstance(want, tuple) and len(want) == 2 and not isinstance(want[0], (int, np.integer))


def _material_json(material: Material):
    return None if material == DEFAULT else [[k.decode("latin-1"), v.decode("latin-1")] for k, v in material.props]


def _material_from(data) -> Material:
    if not data:
        return DEFAULT
    return Material(tuple((k.encode("latin-1"), v.encode("latin-1")) for k, v in data))


# --- colour distance ---------------------------------------------------------------


def _linear(c: np.ndarray) -> np.ndarray:
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def oklab(colour) -> np.ndarray:
    """A colour in OKLab, a perceptual colour space: equal steps look equal."""
    r, g, b = _linear(np.asarray(colour[:3], dtype=np.float64))
    lms = np.array([
        0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b,
        0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b,
        0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b,
    ])
    l_, m_, s_ = np.cbrt(lms)
    return np.array([
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    ])


def from_oklab(lab) -> RGBA:
    """An opaque sRGB colour from OKLab, clipped to the sRGB range."""
    L, a, b = (float(v) for v in lab)
    l_ = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    lin = np.array([
        4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
        -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
        -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_,
    ])
    lin = np.clip(lin, 0.0, 1.0)
    srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
    r, g, b = (int(round(float(v) * 255)) for v in np.clip(srgb, 0.0, 1.0))
    return (r, g, b, 255)


RAMP_SPAN = 0.42  # OKLab lightness from the darkest to the lightest shade
RAMP_LOW, RAMP_HIGH = 0.12, 0.96


def ramp(colour, count: int) -> list[RGBA]:
    """`count` shades of `colour`, dark to light, for a shader that asks for
    a list of colours. The colour itself is one of them; the others step in
    lightness and keep its hue, so the list reads as one material."""
    count = max(1, int(count))
    base = rgba(colour)
    if count == 1:
        return [base]
    L, a, b = oklab(base)
    step = RAMP_SPAN / (count - 1)
    low = min(max(RAMP_LOW, L - RAMP_SPAN / 2), RAMP_HIGH - RAMP_SPAN)
    at = int(round((L - low) / step))
    at = max(0, min(count - 1, at))
    out = []
    for k in range(count):
        if k == at:
            out.append(base)
            continue
        Lk = L + (k - at) * step
        fade = max(0.0, 1.0 - abs(k - at) * 0.12)  # ends lose some colour, as shades do
        out.append(from_oklab((min(max(Lk, 0.0), 1.0), a * fade, b * fade)))
    return out


def nearest(colour, options) -> int | None:
    """The id in `options` ((id, colour) pairs) that looks closest to `colour`."""
    if not options:
        return None
    target = oklab(colour)
    return min(options, key=lambda o: float(np.sum((oklab(o[1]) - target) ** 2)))[0]
