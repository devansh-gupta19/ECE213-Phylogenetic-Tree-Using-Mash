#!/usr/bin/env python3
"""
compare_nwk.py  –  Compare a query Newick tree against a reference Newick tree.

Metrics:
  • Leaf / taxa set comparison
  • Robinson-Foulds (RF) distance  (raw + normalised)
  • Shared / unique bipartitions
  • Strict topology match (unrooted, branch-length agnostic)

No third-party libraries required – pure Python 3.6+.

Usage:
    python compare_nwk.py <query.nwk> <reference.nwk>
"""

import sys
import re
from itertools import combinations


# ──────────────────────────────────────────────
# Newick parser
# ──────────────────────────────────────────────

class Node:
    __slots__ = ("label", "branch_length", "children", "parent")

    def __init__(self):
        self.label: str = ""
        self.branch_length: float = 0.0
        self.children: list = []
        self.parent = None


class NewickTree:
    def __init__(self, root: Node):
        self.root = root

    def leaves(self):
        """Return all leaf nodes."""
        result = []
        stack = [self.root]
        while stack:
            n = stack.pop()
            if not n.children:
                result.append(n)
            else:
                stack.extend(n.children)
        return result

    def leaf_labels(self) -> set:
        return {n.label for n in self.leaves()}


def _parse_newick(s: str, pos: int):
    """Recursive-descent Newick parser. Returns (Node, new_pos)."""
    node = Node()

    # skip whitespace
    while pos < len(s) and s[pos].isspace():
        pos += 1

    # internal node: parse children
    if pos < len(s) and s[pos] == '(':
        pos += 1  # consume '('
        while True:
            while pos < len(s) and s[pos].isspace():
                pos += 1
            child, pos = _parse_newick(s, pos)
            child.parent = node
            node.children.append(child)
            while pos < len(s) and s[pos].isspace():
                pos += 1
            if pos >= len(s):
                raise ValueError("Unexpected end of string inside '()'")
            if s[pos] == ',':
                pos += 1
                continue
            if s[pos] == ')':
                pos += 1
                break
            raise ValueError(f"Expected ',' or ')' at position {pos}, got '{s[pos]}'")

    # optional label
    while pos < len(s) and s[pos].isspace():
        pos += 1
    label_chars = []
    while pos < len(s) and s[pos] not in (':', ',', ')', '(', ';') and not s[pos].isspace():
        label_chars.append(s[pos])
        pos += 1
    node.label = "".join(label_chars)

    # optional branch length
    while pos < len(s) and s[pos].isspace():
        pos += 1
    if pos < len(s) and s[pos] == ':':
        pos += 1
        num_chars = []
        while pos < len(s) and s[pos] not in (',', ')', ';') and not s[pos].isspace():
            num_chars.append(s[pos])
            pos += 1
        try:
            node.branch_length = float("".join(num_chars))
        except ValueError:
            pass  # malformed branch length – ignore

    return node, pos


def parse_newick(text: str) -> NewickTree:
    text = text.strip().rstrip(";")
    root, _ = _parse_newick(text, 0)
    return NewickTree(root)


def read_newick(path: str) -> NewickTree:
    with open(path) as f:
        content = f.read().replace("\n", "").replace("\r", "").strip()
    return parse_newick(content)


# ──────────────────────────────────────────────
# Bipartition (split) extraction
# ──────────────────────────────────────────────

def _collect_splits(node: Node, all_leaves: frozenset, splits: set) -> frozenset:
    """DFS: return the frozenset of leaf-labels in this subtree; record splits."""
    if not node.children:
        return frozenset({node.label})

    sub = frozenset().union(*[_collect_splits(c, all_leaves, splits) for c in node.children])

    # skip trivial splits (size 1) and the root split (whole set)
    if 2 <= len(sub) <= len(all_leaves) - 2:
        complement = all_leaves - sub
        # canonical: store the lexicographically smaller half
        splits.add(min(sub, complement))

    return sub


def get_bipartitions(tree: NewickTree) -> set:
    all_leaves = frozenset(tree.leaf_labels())
    splits = set()
    _collect_splits(tree.root, all_leaves, splits)
    return splits


# ──────────────────────────────────────────────
# Robinson-Foulds distance
# ──────────────────────────────────────────────

def robinson_foulds(q_splits: set, r_splits: set, n_leaves: int) -> dict:
    shared   = q_splits & r_splits
    only_q   = q_splits - r_splits
    only_r   = r_splits - q_splits
    rf       = len(only_q) + len(only_r)
    max_rf   = 2 * (n_leaves - 3) if n_leaves >= 4 else 1
    norm_rf  = rf / max_rf if max_rf > 0 else 0.0
    return {
        "shared":    len(shared),
        "only_q":    len(only_q),
        "only_r":    len(only_r),
        "rf":        rf,
        "norm_rf":   norm_rf,
        "splits_only_q": only_q,
        "splits_only_r": only_r,
    }


# ──────────────────────────────────────────────
# Canonical topology string (unrooted)
# Re-root at the lex-smallest leaf, serialise sorted children.
# ──────────────────────────────────────────────

def _build_adj(node: Node, adj: dict):
    for c in node.children:
        adj.setdefault(id(node), []).append(c)
        adj.setdefault(id(c),    []).append(node)
        _build_adj(c, adj)


def canonical_newick(tree: NewickTree) -> str:
    # build undirected adjacency (by node id)
    nodes_by_id = {}
    adj = {}

    def collect(n):
        nodes_by_id[id(n)] = n
        for c in n.children:
            adj.setdefault(id(n), []).append(id(c))
            adj.setdefault(id(c), []).append(id(n))
            collect(c)

    collect(tree.root)

    # find lex-smallest leaf as re-root point
    leaves = [n for n in nodes_by_id.values() if not n.children]
    start  = min(leaves, key=lambda n: n.label)

    def dfs(nid, parent_id):
        n = nodes_by_id[nid]
        children_ids = [x for x in adj.get(nid, []) if x != parent_id]
        if not children_ids:
            return n.label
        child_strs = sorted(dfs(cid, nid) for cid in children_ids)
        inner = ",".join(child_strs)
        lbl   = n.label or ""
        return f"({inner}){lbl}"

    return dfs(id(start), None)


# ──────────────────────────────────────────────
# Restrict bipartitions to a common leaf subset
# ──────────────────────────────────────────────

def restrict_splits(splits: set, common: frozenset) -> set:
    out = set()
    for s in splits:
        restricted = s & common
        complement = common - restricted
        if len(restricted) >= 2 and len(complement) >= 2:
            out.add(min(restricted, complement))
    return out


# ──────────────────────────────────────────────
# Report helpers
# ──────────────────────────────────────────────

WIDTH = 60

def hr(char="─"):
    print(char * WIDTH)

def section(title):
    print(f"\n[{title}]")

def row(label, value, width=30):
    print(f"  {label:<{width}}: {value}")


def fmt_split(s: frozenset) -> str:
    return "{ " + "  ".join(sorted(s)) + " }"


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <query.nwk> <reference.nwk>")
        sys.exit(1)

    query_path = sys.argv[1]
    ref_path   = sys.argv[2]

    # ── Load trees ────────────────────────────
    try:
        q_tree = read_newick(query_path)
        r_tree = read_newick(ref_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Parse error: {e}")
        sys.exit(1)

    # ── Leaf sets ─────────────────────────────
    q_leaves = q_tree.leaf_labels()
    r_leaves = r_tree.leaf_labels()
    common   = q_leaves & r_leaves
    only_q_l = q_leaves - r_leaves
    only_r_l = r_leaves - q_leaves
    leaf_match = not only_q_l and not only_r_l

    # ── Bipartitions ──────────────────────────
    q_splits_full = get_bipartitions(q_tree)
    r_splits_full = get_bipartitions(r_tree)

    common_fs = frozenset(common)
    if leaf_match:
        q_splits = q_splits_full
        r_splits = r_splits_full
    else:
        q_splits = restrict_splits(q_splits_full, common_fs)
        r_splits = restrict_splits(r_splits_full, common_fs)

    n_common = len(common)
    rf_res   = robinson_foulds(q_splits, r_splits, n_common)

    # ── Topology ──────────────────────────────
    topo_match = canonical_newick(q_tree) == canonical_newick(r_tree)

    # ── Accuracy score ────────────────────────
    if n_common >= 4:
        accuracy = max(0.0, 1.0 - rf_res["norm_rf"]) * 100
    else:
        accuracy = 100.0 if topo_match else 0.0

    # ── Print report ──────────────────────────
    hr("═")
    print("  NWK TREE COMPARISON REPORT")
    hr("═")

    section("Files")
    row("Query",     query_path)
    row("Reference", ref_path)

    section("Leaf / Taxa Summary")
    row("Query leaves",     len(q_leaves))
    row("Reference leaves", len(r_leaves))
    row("Common leaves",    n_common)
    row("Leaf set match",   "YES ✓" if leaf_match else "NO  ✗")
    if only_q_l:
        row("Only in query",   ", ".join(sorted(only_q_l)))
    if only_r_l:
        row("Only in reference", ", ".join(sorted(only_r_l)))

    rf_title = "Robinson-Foulds Distance"
    if not leaf_match:
        rf_title += f" (restricted to {n_common} common leaves)"
    section(rf_title)
    row("Query bipartitions",     len(q_splits))
    row("Reference bipartitions", len(r_splits))
    row("Shared bipartitions",    rf_res["shared"])
    row("Only in query",          rf_res["only_q"])
    row("Only in reference",      rf_res["only_r"])
    hr()
    row("RF distance (raw)",  rf_res["rf"])
    row("Normalised RF",      f"{rf_res['norm_rf']:.4f}  (0=identical, 1=max different)")

    section("Topology")
    row("Strict topology match", "YES ✓" if topo_match else "NO  ✗")

    section("Overall Accuracy Score")
    hr()
    suffix = "  (based on common leaves only)" if not leaf_match else ""
    print(f"  Score: {accuracy:.2f} %{suffix}")
    hr("═")

    # ── Differing bipartitions ────────────────
    only_q_sp = rf_res["splits_only_q"]
    only_r_sp = rf_res["splits_only_r"]
    LIMIT = 10

    if only_q_sp or only_r_sp:
        section("Differing Bipartitions (up to 10 each)")
        if only_q_sp:
            print(f"  Only in query ({len(only_q_sp)} total):")
            for s in list(only_q_sp)[:LIMIT]:
                print(f"    {fmt_split(s)}")
        if only_r_sp:
            print(f"  Only in reference ({len(only_r_sp)} total):")
            for s in list(only_r_sp)[:LIMIT]:
                print(f"    {fmt_split(s)}")

    print()
    # exit 0 if trees are identical, 1 otherwise
    sys.exit(0 if rf_res["rf"] == 0 and leaf_match else 1)


if __name__ == "__main__":
    main()