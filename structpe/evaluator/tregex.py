#!/usr/bin/env python3
"""
Implementation of Tregex-like pattern matching with Lark + NLTK trees.

We unify the approach so that we always build a "grammar string" from the sample
(or pass it in directly), then parse + apply Tregex patterns to find matches.
"""

import importlib
from lark import Lark as LarkParser, Transformer as LarkTransformer, Tree as LarkTree, Token
from nltk.tree import Tree

#############################################
# 1) TREGEX PATTERN GRAMMAR & PARSER
#############################################
TREGEX_PATTERN_GRAMMAR = r"""
    ?pattern: disjunction
    ?disjunction: conjunction ("|" conjunction)*
    ?conjunction: unit ("&" unit)*
    ?unit: "!" unit                -> negation
         | atom
    ?atom: nodepattern (relation pattern)*
         | "(" pattern ")"         -> group

    nodepattern: LABEL ("=" NAME)?  -> nodepattern
    relation: "<" | ">" | "$"
    LABEL: /[^<>&|!()=\s]+/
    NAME: /[A-Za-z_][A-Za-z0-9_]*/
    %import common.WS
    %ignore WS
"""

tregex_parser = LarkParser(
    TREGEX_PATTERN_GRAMMAR,
    parser="lalr",
    start="pattern"
)

#############################################
# 2) TREGEX PATTERN CLASSES
#############################################
class PatternBase:
    parent_map = {}

    def match(self, node, parent):
        raise NotImplementedError

class NodePattern(PatternBase):
    def __init__(self, label, capture=None):
        self.label = label
        self.capture = capture
        self.child_patterns = []
        self.parent_patterns = []
        self.sibling_patterns = []

    def add_child(self, pat):
        self.child_patterns.append(pat)

    def add_parent(self, pat):
        self.parent_patterns.append(pat)

    def add_sibling(self, pat):
        self.sibling_patterns.append(pat)

    def match(self, node, parent):
        from nltk.tree import Tree as NTree

        node_label = node.label() if isinstance(node, NTree) else str(node)
        if self.label != node_label:
            return []

        current_caps = {}
        if self.capture:
            current_caps[self.capture] = node

        # CHILD patterns
        child_results = [current_caps]
        if self.child_patterns:
            if not isinstance(node, NTree):
                return []
            children = list(node)
            if not children:
                return []

            def match_child_patterns(pats, avail_children):
                if not pats:
                    yield {}
                    return
                first, rest = pats[0], pats[1:]
                for i, child in enumerate(avail_children):
                    for child_caps in first.match(child, node):
                        remaining = avail_children[:i] + avail_children[i+1:]
                        for rest_caps in match_child_patterns(rest, remaining):
                            merged = child_caps.copy()
                            conflict = False
                            for k, v in rest_caps.items():
                                if k in merged and merged[k] is not v:
                                    conflict = True
                                    break
                                merged[k] = v
                            if not conflict:
                                yield merged

            new_child_results = []
            for caps_dct in child_results:
                for assignment in match_child_patterns(self.child_patterns, children):
                    merged = caps_dct.copy()
                    conflict = False
                    for k, v in assignment.items():
                        if k in merged and merged[k] is not v:
                            conflict = True
                            break
                        merged[k] = v
                    if not conflict:
                        new_child_results.append(merged)
            if not new_child_results:
                return []
            child_results = new_child_results

        # SIBLING patterns
        sibling_results = []
        if self.sibling_patterns:
            if parent is None or not isinstance(parent, NTree):
                return []
            siblings = [s for s in parent if s is not node]
            if not siblings:
                return []

            def match_sibling_patterns(pats, avail_sibs):
                if not pats:
                    yield {}
                    return
                first, rest = pats[0], pats[1:]
                for i, sib in enumerate(avail_sibs):
                    for sib_caps in first.match(sib, parent):
                        remaining = avail_sibs[:i] + avail_sibs[i+1:]
                        for rest_caps in match_sibling_patterns(rest, remaining):
                            merged = sib_caps.copy()
                            conflict = False
                            for k, v in rest_caps.items():
                                if k in merged and merged[k] is not v:
                                    conflict = True
                                    break
                                merged[k] = v
                            if not conflict:
                                yield merged

            for caps_dct in child_results:
                for assignment in match_sibling_patterns(self.sibling_patterns, siblings):
                    merged = caps_dct.copy()
                    conflict = False
                    for k, v in assignment.items():
                        if k in merged and merged[k] is not v:
                            conflict = True
                            break
                        merged[k] = v
                    if not conflict:
                        sibling_results.append(merged)
            if not sibling_results:
                return []
        else:
            sibling_results = child_results

        # PARENT patterns
        parent_results = []
        if self.parent_patterns:
            if parent is None:
                return []
            grandparent = PatternBase.parent_map.get(id(parent), None)
            for caps_dct in sibling_results:
                merged_caps = caps_dct.copy()
                ok = True
                for patt in self.parent_patterns:
                    sub_matches = patt.match(parent, grandparent)
                    if not sub_matches:
                        ok = False
                        break
                    pick = sub_matches[0]
                    for k, v in pick.items():
                        if k in merged_caps and merged_caps[k] is not v:
                            ok = False
                            break
                        merged_caps[k] = v
                    if not ok:
                        break
                if ok:
                    parent_results.append(merged_caps)
        else:
            parent_results = sibling_results

        return parent_results

class AndPattern(PatternBase):
    def __init__(self, subpatterns):
        self.subpatterns = subpatterns

    def match(self, node, parent):
        results = [{}]
        for sp in self.subpatterns:
            new_results = []
            for partial in results:
                for m in sp.match(node, parent):
                    merged = partial.copy()
                    conflict = False
                    for k, v in m.items():
                        if k in merged and merged[k] is not v:
                            conflict = True
                            break
                        merged[k] = v
                    if not conflict:
                        new_results.append(merged)
            if not new_results:
                return []
            results = new_results
        return results

class OrPattern(PatternBase):
    def __init__(self, alternatives):
        self.alternatives = alternatives

    def match(self, node, parent):
        results = []
        seen = set()
        for alt in self.alternatives:
            for m in alt.match(node, parent):
                key = tuple(sorted((k, id(v)) for k, v in m.items()))
                if key not in seen:
                    seen.add(key)
                    results.append(m)
        return results

class NotPattern(PatternBase):
    def __init__(self, pattern):
        self.pattern = pattern

    def match(self, node, parent):
        if self.pattern.match(node, parent):
            return []
        return [{}]

#############################################
# 3) Lark -> Tregex Transformer
#############################################
class TregexTransformer(LarkTransformer):
    def nodepattern(self, items):
        label = str(items[0])
        capture = str(items[1]) if len(items) > 1 else None
        return NodePattern(label, capture)

    def negation(self, items):
        return NotPattern(items[0])

    def conjunction(self, items):
        if len(items) == 1:
            return items[0]
        return AndPattern(items)

    def disjunction(self, items):
        if len(items) == 1:
            return items[0]
        return OrPattern(items)

    def group(self, items):
        return items[0]

    def atom(self, items):
        base = items[0]
        i = 1
        while i < len(items):
            rel = str(items[i])
            subp = items[i + 1]
            if rel == '<':
                base.add_child(subp)
            elif rel == '>':
                base.add_parent(subp)
            elif rel == '$':
                base.add_sibling(subp)
            i += 2
        return base

#############################################
# 4) Lark -> NLTK
#############################################
def lark_tree_to_nltk(node):
    from nltk.tree import Tree as NTree
    if isinstance(node, Token):
        val = str(node).strip()
        if not val:
            return None
        return val
    elif isinstance(node, LarkTree):
        children = [lark_tree_to_nltk(child) for child in node.children]
        children = [c for c in children if c is not None]
        return NTree(node.data, children)
    return str(node)

#############################################
# 5) find_pattern
#############################################
def find_pattern(nltk_tree, pattern_obj):
    from nltk.tree import Tree as NTree
    PatternBase.parent_map = {}

    def map_parents(n, p):
        PatternBase.parent_map[id(n)] = p
        if isinstance(n, NTree):
            for c in n:
                map_parents(c, n)

    map_parents(nltk_tree, None)
    matches = []

    def traverse(n, p):
        for m in pattern_obj.match(n, p):
            matches.append(m)
        if isinstance(n, NTree):
            for c in n:
                traverse(c, n)

    traverse(nltk_tree, None)
    return matches

#############################################
# 6) parse_and_match
#############################################
def parse_and_match(sample, grammar_module_path, query):
    """
    parse_and_match(sample, grammar_module_path, query) => dict with "queries_results" and a "debug" section.
    You keep 'grammar_text = sample' if you are passing the grammar lines yourself.
    """
    debug_info = {
        "success": False,
        "error": None,
        "grammar_module_path": grammar_module_path,
        "query": query,
        "grammar_text": None
    }
    out = {
        "queries_results": [],
        "debug": debug_info
    }

    # 1) import dataset module
    if not grammar_module_path:
        debug_info["error"] = "No grammar_module_path provided."
        return out

    try:
        mod = importlib.import_module(grammar_module_path)
    except Exception as e:
        debug_info["error"] = f"Cannot import dataset module => {e}"
        return out

    # 2) get grammar from module or get_grammar()
    grammar = getattr(mod, "SAMPLE_GRAMMAR", None)
    if grammar is None and hasattr(mod, "get_grammar"):
        try:
            grammar = mod.get_grammar()
        except Exception as e:
            debug_info["error"] = f"get_grammar => {e}"
            return out
    if not grammar:
        debug_info["error"] = "No grammar found in dataset module."
        return out

    # 3) Instead of build_fn, you said "grammar_text = sample" must remain
    grammar_text = sample  # CHANGED: we do not call build_grammar_string_for_check

    debug_info["grammar_text"] = grammar_text

    # 4) Parse grammar_text -> NLTK
    from lark import Lark
    try:
        parser = Lark(grammar, parser="earley")
        lark_tree = parser.parse(grammar_text)
        nltk_tree = lark_tree_to_nltk(lark_tree)
    except Exception as e:
        debug_info["error"] = f"Parsing grammar_text failed => {e}"
        return out

    # 5) parse Tregex query
    try:
        q_ast = tregex_parser.parse(query)
        q_transformer = TregexTransformer()
        pattern_obj = q_transformer.transform(q_ast)
    except Exception as e:
        debug_info["error"] = f"Parsing Tregex query => {e}"
        return out

    # 6) find_pattern
    try:
        captures_list = find_pattern(nltk_tree, pattern_obj)
    except Exception as e:
        debug_info["error"] = f"Pattern matching error => {e}"
        return out

    # 7) build results
    results = []
    for caps_dct in captures_list:
        if not caps_dct:
            results.append({
                'capture_name': None,
                'node_label': None,
                'node_text': None
            })
        else:
            for c_name, node in caps_dct.items():
                if hasattr(node, "label"):
                    node_label = node.label()
                else:
                    node_label = str(node)

                # CHANGED: unify text capturing
                if hasattr(node, "leaves"):
                    node_text = ' '.join(node.leaves())
                else:
                    node_text = str(node)

                results.append({
                    'capture_name': c_name,
                    'node_label': node_label,
                    'node_text': node_text
                })

    out["queries_results"] = results
    debug_info["success"] = True
    return out

#############################################
# 7) parse_and_match_with_generic_grammar
#############################################
def parse_and_match_with_generic_grammar(sample, grammar_module_path, query):
    """
    parse_and_match_with_generic_grammar(sample, grammar_module_path, query) => list of capture dicts,
    returning them directly (NOT [out]) so Tregex code can iterate captures.
    """
    debug_info = {
        "success": False,
        "error": None,
        "dataset_name": grammar_module_path,
        "query": query,
        "grammar_text": None
    }

    # 1) import dataset module
    try:
        ds_mod = importlib.import_module(grammar_module_path)
    except Exception as e:
        debug_info["error"] = f"Cannot import dataset module => {e}"
        #print(debug_info)
        return []

    # 2) get grammar
    grammar = getattr(ds_mod, "SAMPLE_GRAMMAR", None)
    if not grammar and hasattr(ds_mod, "get_grammar"):
        try:
            grammar = ds_mod.get_grammar()
        except Exception as e:
            debug_info["error"] = f"get_grammar => {e}"
            #print(debug_info)
            return []
    if not grammar:
        debug_info["error"] = "No grammar found in dataset module."
        #print(debug_info)
        return []

    # 3) grammar_text = sample
    # Because you specifically said keep 'grammar_text = sample' approach
    grammar_text = sample
    debug_info["grammar_text"] = grammar_text

    from lark import Lark
    try:
        parser = Lark(grammar, parser="earley")
        lark_tree = parser.parse(grammar_text)
        nltk_tree = lark_tree_to_nltk(lark_tree)
    except Exception as e:
        debug_info["error"] = f"Parsing grammar_text failed => {e}"
        #print(debug_info)
        return []

    # 5) parse Tregex query
    try:
        q_ast = tregex_parser.parse(query)
        q_transformer = TregexTransformer()
        pattern_obj = q_transformer.transform(q_ast)
    except Exception as e:
        debug_info["error"] = f"Parsing Tregex query => {e}"
        #print(debug_info)
        return []

    # 6) find_pattern
    try:
        captures_list = find_pattern(nltk_tree, pattern_obj)
    except Exception as e:
        debug_info["error"] = f"Pattern matching error => {e}"
        #print(debug_info)
        return []

    # 7) return raw list of captures
    results = []
    for caps_dct in captures_list:
        if not caps_dct:
            results.append({
                'capture_name': None,
                'node_label': None,
                'node_text': None
            })
        else:
            for c_name, node in caps_dct.items():
                if c_name is None:
                    c_name = "unlabeled_capture"

                if hasattr(node, "label"):
                    node_label = node.label()
                else:
                    node_label = str(node)

                if hasattr(node, "leaves"):
                    node_text = ' '.join(node.leaves())
                else:
                    node_text = str(node)

                results.append({
                    'capture_name': c_name,
                    'node_label': node_label,
                    'node_text': node_text
                })

    return results
