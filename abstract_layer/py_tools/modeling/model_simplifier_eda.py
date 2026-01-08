#!/usr/bin/env python3
"""
ACTS Model Simplifier using PyEDA with CNF Equivalence-Preserving Simplification

使用 PyEDA 将约束转换为 CNF，然后应用等价化简规则：
1. 子句内去重
2. 永真子句删除 (A ∨ ¬A)
3. 子句吸收 (C1 ⊆ C2 → 删除 C2)

用法：
    python model_simplifier_eda.py input_model.txt output_model.txt [--verbose]
"""

import re
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, FrozenSet, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

try:
    from pyeda.inter import expr, exprvar, And, Or, Not, Implies
    from pyeda.boolalg.expr import Expression
    PYEDA_AVAILABLE = True
except ImportError:
    PYEDA_AVAILABLE = False
    print("Warning: PyEDA not available. Install with: pip install pyeda", file=sys.stderr)

try:
    import ply.lex as lex
    import ply.yacc as yacc
    PLY_AVAILABLE = True
except ImportError:
    PLY_AVAILABLE = False
    print("Warning: PLY not available. Install with: pip install ply", file=sys.stderr)


class ConstraintLexer:
    """PLY Lexer for ACTS constraints"""
    
    tokens = (
        'IDENT', 'STRING',
        'LPAREN', 'RPAREN',
        'AND', 'OR', 'NOT', 'IMPLIES',
        'EQ', 'NEQ',
    )
    
    t_LPAREN = r'\('
    t_RPAREN = r'\)'
    t_ignore = ' \t\n'
    
    # Use functions for multi-char tokens to ensure proper ordering
    def t_IMPLIES(self, t):
        r'=>'
        return t
    
    def t_AND(self, t):
        r'&&'
        return t
    
    def t_OR(self, t):
        r'\|\|'
        return t
    
    def t_NEQ(self, t):
        r'!='
        return t
    
    def t_NOT(self, t):
        r'!'
        return t
    
    def t_EQ(self, t):
        r'='
        return t
    
    def t_STRING(self, t):
        r'"[^"]*"'
        t.value = t.value[1:-1]  # Remove quotes
        return t
    
    def t_IDENT(self, t):
        r'[a-zA-Z_][a-zA-Z0-9_]*'
        return t
    
    def t_error(self, t):
        t.lexer.skip(1)
    
    def __init__(self):
        self.lexer = lex.lex(module=self, debug=False, errorlog=lex.NullLogger())


class ConstraintParser:
    """PLY Parser for ACTS constraints"""
    
    tokens = ConstraintLexer.tokens
    
    # Precedence (lowest to highest)
    precedence = (
        ('left', 'IMPLIES'),
        ('left', 'OR'),
        ('left', 'AND'),
        ('right', 'NOT'),
    )
    
    def __init__(self, variables: Dict[Tuple[str, str], Any]):
        self.variables = variables
        self.lexer = ConstraintLexer()
        # Suppress PLY warnings
        self.parser = yacc.yacc(
            module=self, 
            debug=False, 
            write_tables=False,
            errorlog=yacc.NullLogger()
        )
    
    def parse(self, text: str) -> Optional[Any]:
        try:
            return self.parser.parse(text, lexer=self.lexer.lexer)
        except Exception:
            return None
    
    def p_expr_implies(self, p):
        'expr : expr IMPLIES expr'
        if p[1] is not None and p[3] is not None:
            p[0] = Implies(p[1], p[3])
        else:
            p[0] = p[3]
    
    def p_expr_or(self, p):
        'expr : expr OR expr'
        if p[1] is not None and p[3] is not None:
            p[0] = Or(p[1], p[3])
        elif p[1] is not None:
            p[0] = p[1]
        else:
            p[0] = p[3]
    
    def p_expr_and(self, p):
        'expr : expr AND expr'
        if p[1] is not None and p[3] is not None:
            p[0] = And(p[1], p[3])
        elif p[1] is not None:
            p[0] = p[1]
        else:
            p[0] = p[3]
    
    def p_expr_not(self, p):
        'expr : NOT expr'
        if p[2] is not None:
            p[0] = Not(p[2])
        else:
            p[0] = None
    
    def p_expr_paren(self, p):
        'expr : LPAREN expr RPAREN'
        p[0] = p[2]
    
    def p_expr_eq(self, p):
        'expr : IDENT EQ STRING'
        key = (p[1], p[3])
        if key in self.variables:
            p[0] = self.variables[key]
        else:
            p[0] = None
    
    def p_expr_neq(self, p):
        'expr : IDENT NEQ STRING'
        key = (p[1], p[3])
        if key in self.variables:
            p[0] = Not(self.variables[key])
        else:
            p[0] = None
    
    def p_error(self, p):
        pass


class CNFSimplifier:
    """
    CNF 等价化简器
    
    规则：
    1. 子句内去重: (A ∨ B ∨ A) ≡ (A ∨ B)
    2. 永真子句删除: (A ∨ ¬A ∨ B) ≡ True → 删除整个子句
    3. 子句吸收: C1 ⊆ C2 → 删除 C2
    """
    
    @staticmethod
    def clause_to_set(clause: str) -> Optional[FrozenSet[str]]:
        """将子句字符串转换为文字集合"""
        # 解析 "A || B || C" 格式
        literals = [lit.strip() for lit in clause.split('||')]
        return frozenset(literals) if literals else None
    
    @staticmethod
    def set_to_clause(lit_set: FrozenSet[str]) -> str:
        """将文字集合转换回子句字符串"""
        return ' || '.join(sorted(lit_set, key=lambda x: (x.count('Step'), x)))
    
    @staticmethod
    def is_tautology(lit_set: FrozenSet[str]) -> bool:
        """检查子句是否是永真式 (包含 A 和 ¬A)"""
        for lit in lit_set:
            # 检查 Param="val" 和 Param!="val"
            if '!=' in lit:
                # 找到对应的 Param="val"
                pos_lit = lit.replace('!=', '=')
                if pos_lit in lit_set:
                    return True
            elif '="' in lit and '!="' not in lit:
                # 找到对应的 Param!="val"
                neg_lit = lit.replace('="', '!="')
                if neg_lit in lit_set:
                    return True
        return False
    
    @staticmethod
    def simplify_clauses_batch(clauses: List[str]) -> List[str]:
        """批量简化子句（用于并行）"""
        result = []
        for clause in clauses:
            lit_set = CNFSimplifier.clause_to_set(clause)
            if lit_set is None:
                continue
            # 检查永真式
            if CNFSimplifier.is_tautology(lit_set):
                continue
            result.append((lit_set, clause))
        return result
    
    @staticmethod
    def simplify(clauses: List[str], verbose: bool = False, 
                 max_workers: Optional[int] = None) -> List[str]:
        """
        对 CNF 子句列表进行等价化简
        """
        if not clauses:
            return clauses
        
        print(f"  CNF simplification: {len(clauses)} clauses")
        
        # 第一步：子句内去重 + 永真子句删除（并行）
        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count(), 8)
        
        # 转换为集合表示
        clause_sets: List[FrozenSet[str]] = []
        batch_size = max(1000, len(clauses) // max_workers)
        batches = [clauses[i:i+batch_size] for i in range(0, len(clauses), batch_size)]
        
        tautology_count = 0
        
        if len(batches) > 1 and len(clauses) > 1000:
            print(f"    Phase 1: Intra-clause simplification (parallel, {len(batches)} batches)...")
            try:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(CNFSimplifier.simplify_clauses_batch, b) for b in batches]
                    for future in as_completed(futures):
                        for lit_set, _ in future.result():
                            clause_sets.append(lit_set)
                tautology_count = len(clauses) - len(clause_sets)
            except Exception as e:
                print(f"    Parallel failed, falling back: {e}")
                clause_sets = []
                for clause in clauses:
                    lit_set = CNFSimplifier.clause_to_set(clause)
                    if lit_set and not CNFSimplifier.is_tautology(lit_set):
                        clause_sets.append(lit_set)
                tautology_count = len(clauses) - len(clause_sets)
        else:
            print(f"    Phase 1: Intra-clause simplification...")
            for clause in clauses:
                lit_set = CNFSimplifier.clause_to_set(clause)
                if lit_set is None:
                    continue
                if CNFSimplifier.is_tautology(lit_set):
                    tautology_count += 1
                    continue
                clause_sets.append(lit_set)
        
        if tautology_count > 0:
            print(f"    Removed {tautology_count} tautology clauses")
        
        # 去重
        unique_sets = list(set(clause_sets))
        dedup_count = len(clause_sets) - len(unique_sets)
        if dedup_count > 0:
            print(f"    Removed {dedup_count} duplicate clauses")
        clause_sets = unique_sets
        
        # 第二步：子句吸收 (C1 ⊆ C2 → 删除 C2)
        print(f"    Phase 2: Subsumption elimination...")
        
        # 按子句大小排序（小的在前，更容易吸收大的）
        clause_sets.sort(key=len)
        n = len(clause_sets)
        
        # 使用索引优化：为每个文字建立包含它的子句索引
        lit_to_clauses: Dict[str, Set[int]] = {}
        for i, cs in enumerate(clause_sets):
            for lit in cs:
                if lit not in lit_to_clauses:
                    lit_to_clauses[lit] = set()
                lit_to_clauses[lit].add(i)
        
        absorbed: Set[int] = set()
        
        for i in range(n):
            if i in absorbed:
                continue
            c1 = clause_sets[i]
            if len(c1) == 0:
                continue
            
            # 找所有包含 c1 所有文字的子句（即 c1 的超集）
            # 这些子句的索引是所有文字索引集合的交集
            candidate_sets = [lit_to_clauses.get(lit, set()) for lit in c1]
            if not candidate_sets:
                continue
            
            # 交集操作
            candidates = candidate_sets[0].copy()
            for cs in candidate_sets[1:]:
                candidates &= cs
            
            # c1 ⊆ c2 → 删除 c2
            for j in candidates:
                if j > i and j not in absorbed:
                    c2 = clause_sets[j]
                    if len(c2) > len(c1) and c1.issubset(c2):
                        absorbed.add(j)
        
        result_sets = [cs for i, cs in enumerate(clause_sets) if i not in absorbed]
        absorbed_count = len(absorbed)
        
        if absorbed_count > 0:
            print(f"    Absorbed {absorbed_count} clauses")
        
        # 转换回字符串
        result = [CNFSimplifier.set_to_clause(cs) for cs in result_sets]
        
        print(f"    Final: {len(result)} clauses")
        return result


class EDAModelSimplifier:
    """使用 PyEDA 简化 ACTS 模型约束"""
    
    def __init__(self, parameters: Dict[str, List[str]], verbose: bool = False):
        self.parameters = parameters
        self.verbose = verbose
        # 创建变量映射: (param, value) -> pyeda variable
        self.vars: Dict[Tuple[str, str], Expression] = {}
        self.var_to_pv: Dict[str, Tuple[str, str]] = {}
        
        for param, values in parameters.items():
            for val in values:
                var_name = self._make_var_name(param, val)
                self.vars[(param, val)] = exprvar(var_name)
                self.var_to_pv[var_name] = (param, val)
        
        # 创建 PLY 解析器
        if PLY_AVAILABLE:
            self.ply_parser = ConstraintParser(self.vars)
        else:
            self.ply_parser = None
    
    def _make_var_name(self, param: str, val: str) -> str:
        """创建合法的变量名"""
        # 替换特殊字符
        safe_val = re.sub(r'[^a-zA-Z0-9]', '_', val)
        return f"V_{param}_{safe_val}"
    
    def parse_constraint(self, constraint_str: str) -> Optional[Expression]:
        """将 ACTS 约束解析为 PyEDA 表达式（使用 PLY）"""
        constraint_str = constraint_str.strip()
        if not constraint_str or constraint_str.startswith('['):
            return None
        
        # 使用 PLY 解析器
        if self.ply_parser:
            result = self.ply_parser.parse(constraint_str)
            if result is not None:
                return result
            elif self.verbose:
                print(f"Warning: PLY failed to parse: {constraint_str[:80]}...", file=sys.stderr)
        
        return None
    
    def expr_to_acts(self, e: Expression) -> str:
        """将 PyEDA 表达式转回 ACTS 格式"""
        return self._expr_to_acts_recursive(e)
    
    def _expr_to_acts_recursive(self, e: Expression) -> str:
        """递归转换表达式"""
        name = e.__class__.__name__
        
        if name == 'Variable':
            var_name = str(e)
            if var_name in self.var_to_pv:
                param, val = self.var_to_pv[var_name]
                return f'{param}="{val}"'
            return str(e)
        
        if name == 'Complement':  # Not (literal)
            # 获取对应的变量
            var_name = str(e)[1:]  # 去掉前面的 ~
            if var_name in self.var_to_pv:
                param, val = self.var_to_pv[var_name]
                return f'{param}!="{val}"'
            return f'!({var_name})'
        
        if name == 'NotOp':  # Not (operator)
            inner = self._expr_to_acts_recursive(e.xs[0])
            m = re.match(r'^(\w+)="([^"]+)"$', inner)
            if m:
                return f'{m.group(1)}!="{m.group(2)}"'
            return f'!({inner})'
        
        if name == 'OrOp':  # Or
            parts = [self._expr_to_acts_recursive(x) for x in e.xs]
            return ' || '.join(parts)
        
        if name == 'AndOp':  # And
            parts = [self._expr_to_acts_recursive(x) for x in e.xs]
            return ' && '.join(f'({p})' if ' || ' in p else p for p in parts)
        
        # 其他情况，尝试遍历 xs
        if hasattr(e, 'xs') and e.xs:
            parts = [self._expr_to_acts_recursive(x) for x in e.xs]
            return ' && '.join(parts)
        
        return str(e)
    
    def simplify_constraints(self, constraints: List[str], 
                              max_workers: Optional[int] = None) -> List[str]:
        """简化约束列表"""
        print(f"Parsing {len(constraints)} constraints...")
        
        # 解析所有约束
        parsed = []
        for c in constraints:
            e = self.parse_constraint(c)
            if e is not None:
                parsed.append((c, e))
        
        print(f"  Parsed {len(parsed)} valid constraints")
        
        if not parsed:
            return constraints
        
        # 合并为一个大合取式
        print("Combining constraints into conjunction...")
        all_exprs = [e for _, e in parsed]
        combined = And(*all_exprs) if len(all_exprs) > 1 else all_exprs[0]
        
        # 转换为 CNF
        print("Converting to CNF...")
        try:
            cnf = combined.to_cnf()
            # 注意: PyEDA 的 simplify() 可能会丢失约束，暂时禁用
            # simplified = cnf.simplify()
            simplified = cnf  # 不使用 PyEDA 的 simplify
        except Exception as e:
            print(f"Warning: CNF conversion failed: {e}", file=sys.stderr)
            return constraints
        
        # 提取子句
        print("Extracting clauses...")
        clauses = []
        
        # CNF 是 And of Or 子句
        if hasattr(simplified, 'xs'):  # And node
            for clause in simplified.xs:
                acts_str = self.expr_to_acts(clause)
                clauses.append(acts_str)
        else:
            # 单个表达式
            acts_str = self.expr_to_acts(simplified)
            clauses.append(acts_str)
        
        print(f"  Extracted {len(clauses)} clauses")
        
        # 应用 CNF 等价化简规则
        result = CNFSimplifier.simplify(clauses, self.verbose, max_workers)
        
        return result


def parse_model(model_file: str) -> Tuple[str, Dict[str, List[str]], List[str]]:
    """解析 ACTS 模型文件"""
    with open(model_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    system_name = "ACTSModel"
    parameters: Dict[str, List[str]] = {}
    constraints: List[str] = []
    section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line == "[System]":
            section = "system"
        elif line == "[Parameter]":
            section = "parameter"
        elif line == "[Constraint]":
            section = "constraint"
        elif line in ["[Relation]", "[Misc]"]:
            section = None
        elif section == "system" and line.startswith("Name:"):
            system_name = line.split(":", 1)[1].strip()
        elif section == "parameter":
            match = re.match(r'(\w+)\s*\(enum\)\s*:\s*(.+)', line)
            if match:
                parameters[match.group(1)] = [v.strip() for v in match.group(2).split(',')]
        elif section == "constraint":
            constraints.append(line)
    
    return system_name, parameters, constraints


def write_model(output_file: str, system_name: str, 
                parameters: Dict[str, List[str]], constraints: List[str]):
    """写入 ACTS 模型文件"""
    lines = ["[System]", f"Name: {system_name}", "", "[Parameter]"]
    
    # 排序参数
    step_params = {int(re.match(r'Step(\d+)', k).group(1)): (k, v) 
                   for k, v in parameters.items() if k.startswith('Step') and re.match(r'Step(\d+)', k)}
    other_params = {k: v for k, v in parameters.items() if not k.startswith('Step')}
    
    for n in sorted(step_params.keys()):
        k, v = step_params[n]
        lines.append(f'{k} (enum) : {",".join(v)}')
    for k in sorted(other_params.keys()):
        lines.append(f'{k} (enum) : {",".join(other_params[k])}')
    
    lines.extend(["", "[Relation]", "", "[Constraint]"])
    lines.extend(constraints)
    lines.extend(["", "[Misc]"])
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description="ACTS Model Simplifier using PyEDA")
    parser.add_argument("input_file", help="输入 ACTS 模型文件")
    parser.add_argument("output_file", help="输出 ACTS 模型文件")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    parser.add_argument("--max-workers", type=int, help="并行工作线程数")
    args = parser.parse_args()
    
    if not PYEDA_AVAILABLE:
        print("Error: PyEDA is required. Install with: pip install pyeda", file=sys.stderr)
        sys.exit(1)
    
    if not Path(args.input_file).exists():
        print(f"Error: File not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Reading model: {args.input_file}")
    system_name, parameters, constraints = parse_model(args.input_file)
    print(f"  Parameters: {len(parameters)}, Constraints: {len(constraints)}")
    
    simplifier = EDAModelSimplifier(parameters, args.verbose)
    simplified = simplifier.simplify_constraints(constraints, args.max_workers)
    
    write_model(args.output_file, system_name, parameters, simplified)
    print(f"Written to: {args.output_file}")
    
    reduction = len(constraints) - len(simplified)
    pct = 100 * reduction / len(constraints) if constraints else 0
    print(f"Reduction: {len(constraints)} -> {len(simplified)} (-{reduction}, {pct:.1f}%)")


if __name__ == "__main__":
    main()

