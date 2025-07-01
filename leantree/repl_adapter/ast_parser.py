import string
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

from leantree import utils


class LeanASTNode(ABC):
    @abstractmethod
    def get_children(self) -> "list[LeanASTNode]":
        pass

    def traverse_preorder(self, consumer: "Callable[[LeanASTNode], None]"):
        consumer(self)
        for child in self.get_children():
            child.traverse_preorder(consumer)

    def find_first_node(self, predicate: "Callable[[LeanASTNode], bool]") -> "Optional[LeanASTNode]":
        if predicate(self):
            return self
        for child in self.get_children():
            if result := child.find_first_node(predicate):
                return result

    def get_tokens(self) -> list[str]:
        def visitor(n: LeanASTNode):
            if isinstance(n, LeanASTLiteral):
                tokens.append(n.pretty_print())

        tokens = []
        self.traverse_preorder(visitor)
        return tokens

    def pretty_print(self) -> str:
        def get_children(n: LeanASTNode) -> list[LeanASTNode]:
            return [c for c in n.get_children() if not isinstance(c, LeanASTLiteral)]

        def get_node_label(n: LeanASTNode) -> str:
            if isinstance(n, LeanASTObject):
                return n.type + " " + " ".join(c.value if isinstance(c, LeanASTLiteral) else "_" for c in n.args)
            if isinstance(n, LeanASTArray):
                return "[" + " ".join(c.value if isinstance(c, LeanASTLiteral) else "_" for c in n.items) + "]"
            if isinstance(n, LeanASTLiteral):
                return n.value
            raise Exception("Unknown AST node type.")

        return utils.pretty_print_tree(self, get_children, get_node_label)


@dataclass
class LeanASTObject(LeanASTNode):
    type: str
    args: list[LeanASTNode]

    def get_children(self) -> list[LeanASTNode]:
        return self.args


@dataclass
class LeanASTArray(LeanASTNode):
    items: list[LeanASTNode]

    def get_children(self) -> list[LeanASTNode]:
        return self.items


@dataclass
class LeanASTLiteral(LeanASTNode):
    value: str

    def get_children(self) -> list[LeanASTNode]:
        return []

    def pretty_print(self) -> str:
        if len(self.value) >= 2 and self.value[0] == self.value[-1] == "\"":
            return self.value[1:-1]
        if self.value.startswith("`"):
            return self.value[1:]
        return self.value


@dataclass
class LeanAST:
    root: LeanASTNode

    def traverse_preorder(self, consumer: Callable[[LeanASTNode], None]):
        return self.root.traverse_preorder(consumer)

    def get_tokens(self) -> list[str]:
        return self.root.get_tokens()

    def find_first_node(self, predicate: Callable[[LeanASTNode], bool]) -> Optional[LeanASTNode]:
        return self.root.find_first_node(predicate)

    def pretty_print(self) -> str:
        return self.root.pretty_print()

    @classmethod
    def parse_from_string(cls, s: str) -> "LeanAST":
        def skip_to_argument_end(start_idx: int) -> int:
            assert s[start_idx] not in [*string.whitespace, *"()[]"], f"Argument started with '{s[start_idx]}'"
            if s[start_idx:].startswith("`[anonymous]"):
                return start_idx + len("`[anonymous]")
            if s[start_idx] == "\"":
                i = start_idx + 1
                escaped = False
                while True:
                    if not escaped:
                        if s[i] == "\"":
                            break
                        elif s[i] == "\\":
                            escaped = True
                    else:
                        escaped = False

                    i += 1
                return i + 1
            i = start_idx
            # Addresses arguments like «term(↑)»
            depths = {b: 0 for b in ["«»"]}
            while not (s[i] in [*string.whitespace, *")]"] and all(d == 0 for d in depths.values())):
                for b in depths:
                    if s[i] == b[0]:
                        depths[b] += 1
                    elif s[i] == b[1]:
                        assert depths[b] > 0
                        depths[b] -= 1
                i += 1
            return i

        def skip_whitespaces(start_idx: int) -> int:
            i = start_idx
            while s[i] in string.whitespace:
                i += 1
            return i

        def read_node(start_idx: int) -> tuple[LeanASTNode, int]:
            i = skip_whitespaces(start_idx)
            if s[i] == "(":
                return read_subtree(i)
            if s[i] == "[":
                return read_array(i)
            end_idx = skip_to_argument_end(i)
            assert end_idx > i
            return LeanASTLiteral(s[i:end_idx]), end_idx

        def read_subtree(start_idx: int) -> tuple[LeanASTObject, int]:
            assert s[start_idx] == "("
            type_end = skip_to_argument_end(start_idx + 1)
            node_type = s[start_idx + 1:type_end]
            args = []
            i = type_end
            while s[i] != ")":
                i = skip_whitespaces(i)
                arg, i = read_node(i)
                args.append(arg)
            return LeanASTObject(node_type, args), i + 1

        def read_array(start_idx: int) -> tuple[LeanASTArray, int]:
            assert s[start_idx] == "["
            items = []
            i = start_idx + 1
            while s[i] != "]":
                i = skip_whitespaces(i)
                item, i = read_node(i)
                items.append(item)
            return LeanASTArray(items), i + 1

        s = s.strip()
        root, final_idx = read_node(0)
        assert isinstance(root, LeanASTNode)
        assert final_idx == len(s)
        return LeanAST(root)


ast_str = """
(Command.declaration
 (Command.declModifiers [] [] [] [] [] [])
 (Command.theorem
  "theorem"
  (Command.declId `my_le_total [])
  (Command.declSig
   [(Term.explicitBinder "(" [`x `y] [":" `Nat] [] ")")]
   (Term.typeSpec ":" («term_∨_» («term_≤_» `x "≤" `y) "∨" («term_≤_» `y "≤" `x))))
  (Command.declValSimple
   ":="
   (Term.byTactic
    "by"
    (Tactic.tacticSeq
     (Tactic.tacticSeq1Indented
      [(Tactic.induction "induction" [`x] [] ["generalizing" [`y]] [])
       []
       (Tactic.case
        "case"
        [(Tactic.caseArg (Lean.binderIdent `zero) [])]
        "=>"
        (Tactic.tacticSeq
         (Tactic.tacticSeq1Indented
          [(Tactic.exact "exact" (Term.app `Or.inl [(Term.paren "(" (Term.app `Nat.zero_le [`y]) ")")]))])))
       []
       (Tactic.case
        "case"
        [(Tactic.caseArg (Lean.binderIdent `succ) [(Lean.binderIdent `x) (Lean.binderIdent `ih)])]
        "=>"
        (Tactic.tacticSeq
         (Tactic.tacticSeq1Indented
          [(Tactic.cases
            "cases"
            [(Tactic.casesTarget [] `y)]
            []
            [(Tactic.inductionAlts
              "with"
              []
              [(Tactic.inductionAlt
                [(Tactic.inductionAltLHS "|" (group [] `zero) [])]
                "=>"
                (Tactic.tacticSeq
                 (Tactic.tacticSeq1Indented
                  [(Tactic.exact
                    "exact"
                    (Term.app
                     `Or.inr
                     [(Term.paren
                       "("
                       (Term.app `Nat.zero_le [(Term.paren "(" («term_+_» `x "+" (num "1")) ")")])
                       ")")]))])))
               (Tactic.inductionAlt
                [(Tactic.inductionAltLHS "|" (group [] `succ) [`y'])]
                "=>"
                (Tactic.tacticSeq
                 (Tactic.tacticSeq1Indented
                  [(Tactic.cases
                    "cases"
                    [(Tactic.casesTarget [] (Term.app `ih [`y']))]
                    []
                    [(Tactic.inductionAlts
                      "with"
                      []
                      [(Tactic.inductionAlt
                        [(Tactic.inductionAltLHS "|" (group [] `inl) [`x_le_y'])]
                        "=>"
                        (Tactic.tacticSeq
                         (Tactic.tacticSeq1Indented
                          [(Tactic.exact
                            "exact"
                            (Term.app `Or.inl [(Term.paren "(" (Term.app `Nat.succ_le_succ [`x_le_y']) ")")]))])))
                       (Tactic.inductionAlt
                        [(Tactic.inductionAltLHS "|" (group [] `inr) [`y'_le_x])]
                        "=>"
                        (Tactic.tacticSeq
                         (Tactic.tacticSeq1Indented
                          [(Tactic.exact
                            "exact"
                            (Term.app
                             `Or.inr
                             [(Term.paren "(" (Term.app `Nat.succ_le_succ [`y'_le_x]) ")")]))])))])])])))])])])))])))
   (Termination.suffix [] [])
   [])))
"""

ast_str2 = """
(Command.declaration
 (Command.declModifiers
  [(Command.docComment
    "/--"
    "If `u` is a solution to `E` and `init` designates its first `E.order` values,\n  then `∀ n, u n = E.mkSol init n`. -/")]
  []
  []
  []
  []
  [])
 (Command.theorem
  "theorem"
  (Command.declId `eq_mk_of_is_sol_of_eq_init [])
  (Command.declSig
   [(Term.implicitBinder "{" [`u] [":" (Term.arrow (termℕ "ℕ") "→" `α)] "}")
    (Term.implicitBinder "{" [`init] [":" (Term.arrow (Term.app `Fin [`E.order]) "→" `α)] "}")
    (Term.explicitBinder "(" [`h] [":" (Term.app `E.IsSolution [`u])] [] ")")
    (Term.explicitBinder
     "("
     [`heq]
     [":"
      (Term.forall
       "∀"
       [`n]
       [(Term.typeSpec ":" (Term.app `Fin [`E.order]))]
       ","
       («term_=_» (Term.app `u [`n]) "=" (Term.app `init [`n])))]
     []
     ")")]
   (Term.typeSpec ":" (Term.forall "∀" [`n] [] "," («term_=_» (Term.app `u [`n]) "=" (Term.app `E.mkSol [`init `n])))))
  (Command.declValSimple
   ":="
   (Term.byTactic
    "by"
    (Tactic.tacticSeq
     (Tactic.tacticSeq1Indented
      [(Tactic.intro "intro" [`n])
       []
       (Tactic.rwSeq "rw" (Tactic.optConfig []) (Tactic.rwRuleSeq "[" [(Tactic.rwRule [] `mkSol)] "]") [])
       []
       (Mathlib.Tactic.splitIfs "split_ifs" [] ["with" [(Lean.binderIdent `h')]])
       []
       (Lean.cdot
        (Lean.cdotTk (patternIgnore (token.«· » "·")))
        (Tactic.tacticSeq
         (Tactic.tacticSeq1Indented
          [(Tactic.exact
            "exact"
            (Lean.modCast "mod_cast" (Term.app `heq [(Term.anonymousCtor "⟨" [`n "," `h'] "⟩")])))])))
       []
       (Tactic.rwSeq
        "rw"
        (Tactic.optConfig [])
        (Tactic.rwRuleSeq
         "["
         [(Tactic.rwRule
           [(patternIgnore (token.«← » "←"))]
           (Term.app `tsub_add_cancel_of_le [(Term.paren "(" (Term.app `le_of_not_lt [`h']) ")")]))
          ","
          (Tactic.rwRule [] (Term.app `h [(Term.paren "(" («term_-_» `n "-" `E.order) ")")]))]
         "]")
        [])
       []
       (Batteries.Tactic.congrConfigWith "congr" [] [] "with" [(Tactic.rintroPat.one (Tactic.rcasesPat.one `k))] [])
       []
       (Tactic.tacticHave_
        "have"
        (Term.haveDecl
         (Term.haveIdDecl
          (Term.haveId (hygieneInfo `[anonymous]))
          []
          [(Term.typeSpec ":" («term_<_» («term_+_» («term_-_» `n "-" `E.order) "+" `k) "<" `n))]
          ":="
          (Term.byTactic
           "by"
           (Tactic.tacticSeq (Tactic.tacticSeq1Indented [(Tactic.omega "omega" (Tactic.optConfig []))]))))))
       []
       (Tactic.rwSeq
        "rw"
        (Tactic.optConfig [])
        (Tactic.rwRuleSeq
         "["
         [(Tactic.rwRule
           []
           (Term.app
            `eq_mk_of_is_sol_of_eq_init
            [`h `heq (Term.paren "(" («term_+_» («term_-_» `n "-" `E.order) "+" `k) ")")]))]
         "]")
        [])
       []
       (Tactic.simp "simp" (Tactic.optConfig []) [] [] [] [])])))
   (Termination.suffix [] [])
   [])))
"""

ast_str3 = """
(lemma
 (Command.declModifiers [] [] [] [] [] [])
 (group
  "lemma"
  (Command.declId `nat_pow_one_sub_dvd_pow_mul_sub_one [])
  (Command.declSig
   [(Term.explicitBinder "(" [`x `m `n] [":" (termℕ "ℕ")] [] ")")]
   (Term.typeSpec
    ":"
    («term_∣_»
     («term_-_» («term_^_» `x "^" `m) "-" (num "1"))
     "∣"
     («term_-_» («term_^_» `x "^" (Term.paren "(" («term_*_» `m "*" `n) ")")) "-" (num "1")))))
  (Command.declValSimple
   ":="
   (Term.byTactic
    "by"
    (Tactic.tacticSeq
     (Tactic.tacticSeq1Indented
      [(Mathlib.Tactic.nthRwSeq
        "nth_rw"
        (Tactic.optConfig [])
        (group)
        [(num "2")]
        (Tactic.rwRuleSeq "[" [(Tactic.rwRule [(patternIgnore (token.«← » "←"))] (Term.app `Nat.one_pow [`n]))] "]")
        [])
       []
       (Tactic.rwSeq
        "rw"
        (Tactic.optConfig [])
        (Tactic.rwRuleSeq "[" [(Tactic.rwRule [] (Term.app `Nat.pow_mul [`x `m `n]))] "]")
        [])
       []
       (Tactic.apply
        "apply"
        (Term.app `nat_sub_dvd_pow_sub_pow [(Term.paren "(" («term_^_» `x "^" `m) ")") (num "1")]))])))
   (Termination.suffix [] [])
   [])))
""".strip()

ast_str4 = """
(Tactic.exact
 "exact"
 (Term.app
  `congr_arg
  [(Term.typeAscription "(" (Lean.Elab.Term.CoeImpl.«term(↑)» "(" "↑" ")") ":" [(Term.arrow `Num "→" (termℕ "ℕ"))] ")")
   (Term.paren "(" (Term.app `decode_encodeNum [`n]) ")")]))
""".strip()
# LeanAST.parse_from_string(ast_str)
