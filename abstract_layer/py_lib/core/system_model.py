from typing import List, Literal
import os


class Parameter:
    def __init__(self, name: str, ptype: str, values: List[str]):
        self.name = name
        self.ptype = ptype 
        self.values = values

    def __str__(self):
        return f'{self.name} ({self.ptype}): {", ".join(self.values)}'


class Constraint:
    def __init__(self, expression: str):
        self.expression = expression

    def __str__(self):
        return self.expression


class Relation:
    def __init__(self, name: str, params: List[str], strength: int):
        self.name = name
        self.params = params
        self.strength = strength

    def __str__(self):
        return f'{self.name}: ({", ".join(self.params)}, {self.strength})'


class SystemModel:
    def __init__(self, name: str, strength: int = 1):
        self.name = name
        self.strength = strength
        self.parameters: List[Parameter] = []
        self.constraints: List[Constraint] = []
        self.relations: List[Relation] = []

    def add_parameter(self, name: str, ptype: str, values: List[str]):
        self.parameters.append(Parameter(name, ptype, values))

    def add_constraint(self, expression: str):
        self.constraints.append(Constraint(expression))

    def add_relation(self, name: str, params: List[str], strength: int):
        self.relations.append(Relation(name, params, strength))

    def to_config(self) -> str:
        lines = []
        # System
        lines.append("[System]")
        lines.append(f"Name: {self.name}")
        lines.append(f"Strength: {self.strength}")
        lines.append("")

        # Parameter
        lines.append("[Parameter]")
        for p in self.parameters:
            lines.append(str(p))
        lines.append("")

        # Constraint
        if self.constraints:
            lines.append("[Constraint]")
            for c in self.constraints:
                lines.append(str(c))
            lines.append("")

        # Relation
        if self.relations:
            lines.append("[Relation]")
            for r in self.relations:
                lines.append(str(r))
            lines.append("")

        return "\n".join(lines)

    def save(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_config())
            
    def solve(self, model_path, result_path, algorithm: Literal['ipog', 'ipog_d', 'ipof', 'ipof2', 'basechoice'], use_forbidden_table: bool = True):
        '''
        ACTS Version: 3.2
        Usage: java [options] -jar jarName <inputFileName> [outputFileName]
        where options include:
        -Dalgo=ipog|ipog_d|ipof|ipof2|basechoice|null
                ipog - use algorithm IPO (default)
                ipog_d - use algorithm IPO + Binary Construction (for large systems)
                ipof - use the IPOF method
                ipof2 - use the IPOF2 method
                basechoice - use Base Choice method
                null - use to check coverage only (no test generation)
        -Ddoi=<int>
                specify the degree of interactions to be covered. Use -1 for mixed strength. (default value is 2)
        -Doutput=numeric|nist|csv|excel
                numeric - output test set in numeric format
                nist - output test set in NIST format (default)
                csv - output test set in CSV format
                excel - output test set in EXCEL format
        -Dmode=scratch|extend
                scratch - generate tests from scratch (default)
                extend - extend from an existing test set
        -Dchandler=no|solver|forbiddentuples
                no - ignore all constraints
                solver - handle constraints using CSP solver
                forbiddentuples - handle constraints using minimum forbidden tuples (default)
        -Dcheck=on|off
                on - verify coverage after test generation
                off - do not verify coverage (default)
        -Dprogress=on|off
                on - display progress information
                off - do not display progress information (default)
        -Ddebug=on|off
                on - display debug info
                off - do not display debug info (default)
        -Drandstar=on|off
                on - randomize don't care values (default)
                off - do not randomize don't care values
        -Dcombine=<all>
                all - every possible combination of parameters
        '''
        
        self.save(model_path)
        # Get the jar file path relative to the project root
        jar_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'ct_model', 'tools', 'acts_3.2.jar')
        if use_forbidden_table:
            os.system(f"java -Dprogress=on -Doutput=csv -Ddoi={self.strength} -Dalgo={algorithm} -Dchandler=forbiddentuples -jar {jar_path} {model_path} {result_path}")
        else:
            os.system(f"java -Dprogress=on -Doutput=csv -Ddoi={self.strength} -Dalgo={algorithm} -Dchandler=solver -jar {jar_path} {model_path} {result_path}")