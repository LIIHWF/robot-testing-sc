%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
/**
 * Weakest Precondition (WP) Computation for INDIGOLOG Programs
 * 
 * Part of the INDIGOLOG system
 * 
 * This module implements weakest precondition computation for Golog/INDIGOLOG
 * programs, combining regression computation and poss preconditions.
 * 
 * The weakest precondition WP(E, Q) is the weakest condition that must hold
 * initially for program E to execute and satisfy postcondition Q upon termination.
 * 
 * WP COMPUTATION RULES
 * ====================
 * 
 * | Program Form      | WP Formula                                    |
 * |-------------------|-----------------------------------------------|
 * | []                | Q                                             |
 * | ?(P)              | P ∧ Q                                         |
 * | [E1, E2, ...]     | WP(E1, WP([E2, ...], Q))                     |
 * | if(P, E1, E2)     | (P → WP(E1, Q)) ∧ (¬P → WP(E2, Q))           |
 * | ndet(E1, E2)      | WP(E1, Q) ∨ WP(E2, Q)                        |
 * | while(P, E)       | Iterative approximation                       |
 * | A (primitive)     | poss(A) ∧ regress(Q, A)                      |
 * | proc call         | WP(proc_body, Q)                             |
 * 
 * Usage:
 *   wp(Program, Postcondition, WeakestPrecondition)
 *   
 * Example:
 *   wp([put(bread, plate)], loc(bread, plate), WP)
 */
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  MULTIFILE DECLARATIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- multifile(poss/2).
:- multifile(prim_action/1).
:- multifile(causes_val/4).
:- multifile(causes_true/3).
:- multifile(causes_false/3).
:- multifile(proc/2).
:- multifile(initially/2).
:- multifile(rel_fluent/1).
:- multifile(fun_fluent/1).
:- multifile(senses/2).
:- multifile(senses/5).
:- multifile(system_action/1).
:- multifile(max_proc_expansion/2).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  MAIN WP COMPUTATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/**
 * wp(+Program, +Postcondition, -WP)
 * 
 * Compute the weakest precondition for Program to satisfy Postcondition.
 * Automatically expands proc definitions and simplifies the result.
 * 
 * Proc expansion respects max_proc_expansion/2 limits defined by user.
 */
wp(E, Q, WP) :-
    wp_core(E, Q, WP0),
    simplify_formula(WP0, WP1),
    expand_procs(WP1, WP2),
    simplify_formula(WP2, WP).

/**
 * wp_no_expand(+Program, +Postcondition, -WP)
 * 
 * Compute WP without expanding proc definitions.
 * Use when you want to keep proc calls in the result for readability.
 */
wp_no_expand(E, Q, WP) :-
    wp_core(E, Q, WP0),
    simplify_formula(WP0, WP).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  CORE WP COMPUTATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Empty program: WP([], Q) = Q
wp_core([], Q, Q) :- !.

% Test action: WP(?(P), Q) = P ∧ Q
wp_core(?(P), Q, and(P, Q)) :- !.

% Sequence: WP([E|L], Q) = WP(E, WP(L, Q))
wp_core([E|L], Q, WP) :- !,
    wp_core(L, Q, WP_L),
    wp_core(E, WP_L, WP).

% Non-deterministic choice: WP(ndet(E1, E2), Q) = WP(E1, Q) ∨ WP(E2, Q)
wp_core(ndet(E1, E2), Q, or(WP1, WP2)) :- !,
    wp_core(E1, Q, WP1),
    wp_core(E2, Q, WP2).

% Conditional: WP(if(P, E1, E2), Q) = (P → WP(E1, Q)) ∧ (¬P → WP(E2, Q))
wp_core(if(P, E1, E2), Q, and(impl(P, WP1), impl(neg(P), WP2))) :- !,
    wp_core(E1, Q, WP1),
    wp_core(E2, Q, WP2).

% While loop: Iterative approximation
wp_core(while(P, E), Q, WP) :- !,
    wp_while(P, E, Q, WP).

% Star (Kleene closure): Can terminate or execute once then repeat
wp_core(star(E), Q, WP) :- !,
    wp_core([], Q, WP_Term),
    wp_core(E, WP, WP_Exec),
    WP = or(WP_Term, WP_Exec).

% Bounded star: Execute E exactly N times
wp_core(star(E, N), Q, WP) :- !,
    N > 0,
    N1 is N - 1,
    (N1 = 0 -> L = [] ; L = [star(E, N1)]),
    wp_core([E|L], Q, WP).

% Existential quantification: WP(pi(V, E), Q) = ∃V.WP(E, Q)
wp_core(pi(V, E), Q, some(V, WP_E)) :- !,
    wp_core(E, Q, WP_E).

wp_core(pi((V, D), E), Q, some((V, D), WP_E)) :- !,
    wp_core(E, Q, WP_E).

wp_core(pi([], E), Q, WP) :- !,
    wp_core(E, Q, WP).

wp_core(pi([V|L], E), Q, WP) :- !,
    wp_core(pi(L, pi(V, E)), Q, WP).

% Procedure call: Expand and compute WP
wp_core(E, Q, WP) :-
    proc(E, Body), !,
    wp_core(Body, Q, WP).

% Primitive action: WP(A, Q) = poss(A) ∧ regress(Q, A)
wp_core(A, Q, WP) :-
    (system_action(A) ->
        WP = Q
    ;
        calc_arg(A, A1, []),
        (prim_action(A1) ->
            (poss(A1, P) ->
                regress(A1, Q, Q_Reg),
                WP = and(P, Q_Reg)
            ;
                regress(A1, Q, WP)
            )
        ;
            WP = and(A, Q)
        )
    ).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  WHILE LOOP APPROXIMATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/**
 * wp_while(+P, +E, +Q, -WP)
 * 
 * Approximate WP for while loops using iterative approximation:
 *   WP_0 = ¬P ∧ Q
 *   WP_{i+1} = (¬P ∧ Q) ∨ (P ∧ WP(E, WP_i))
 */
wp_while(P, E, Q, WP) :-
    WP0 = and(neg(P), Q),
    wp_while_iter(P, E, Q, WP0, WP, 10).

wp_while_iter(_, _, _, WP, WP, 0) :- !.
wp_while_iter(P, E, Q, WP_Old, WP_Final, N) :-
    N > 0,
    wp_core(E, WP_Old, WP_E),
    WP_New = or(and(neg(P), Q), and(P, WP_E)),
    (WP_Old == WP_New ->
        WP_Final = WP_New
    ;
        N1 is N - 1,
        wp_while_iter(P, E, Q, WP_New, WP_Final, N1)
    ).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  REGRESSION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/**
 * regress(+Action, +Formula, -RegressedFormula)
 * 
 * Regress Formula through Action using successor state axioms.
 * Replaces fluents with their values before Action was executed.
 */
regress(A, F, R) :-
    regress_impl(F, A, R).

% Logical connectives
regress_impl(and(P1, P2), A, and(R1, R2)) :- !,
    regress_impl(P1, A, R1),
    regress_impl(P2, A, R2).

regress_impl(or(P1, P2), A, or(R1, R2)) :- !,
    regress_impl(P1, A, R1),
    regress_impl(P2, A, R2).

regress_impl(neg(P), A, neg(R)) :- !,
    regress_impl(P, A, R).

regress_impl(impl(P1, P2), A, impl(R1, R2)) :- !,
    regress_impl(P1, A, R1),
    regress_impl(P2, A, R2).

% Quantifiers
regress_impl(some(V, P), A, some(V, R)) :- !,
    regress_impl(P, A, R).

regress_impl(all(V, P), A, all(V, R)) :- !,
    regress_impl(P, A, R).

regress_impl(some((V, D), P), A, some((V, D), R)) :- !,
    regress_impl(P, A, R).

regress_impl(all((V, D), P), A, all((V, D), R)) :- !,
    regress_impl(P, A, R).

% Equality
regress_impl(F = V, A, R) :- !,
    regress_equality(F, V, A, R).

% Proc definitions (expand simple non-recursive ones)
regress_impl(F, A, R) :-
    proc(F, Body),
    is_simple_proc(F), !,
    regress_impl(Body, A, R).

% Relational fluent
regress_impl(F, A, R) :-
    rel_fluent(F), !,
    regress_rel_fluent(F, A, R).

% Base case: not a fluent, keep as is
regress_impl(F, _, F).

% Simple proc check (non-recursive procs like 'empty', 'accessible')
% These procs should be expanded during regression so their fluents get properly regressed
% If a proc contains fluents that need to be regressed, add it here!
is_simple_proc(empty(_)) :- !.
is_simple_proc(accessible(_)) :- !.
is_simple_proc(all_inside_heatable(_)) :- !.
is_simple_proc(some_inside_requires_heat(_)) :- !.
is_simple_proc(in(_, _)) :- !.  % in(O1, O2) expands to transitive closure of loc

/**
 * regress_rel_fluent(+Fluent, +Action, -Result)
 * 
 * Regress a relational fluent through an action.
 * Uses successor state axiom: F holds after A iff
 *   (A makes F true) ∨ (F held before ∧ A doesn't make F false)
 */
regress_rel_fluent(F, A, R) :-
    findall(C, causes_val(A, F, true, C), TrueConds),
    findall(C, causes_val(A, F, false, C), FalseConds),
    (TrueConds = [], FalseConds = [] ->
        R = F  % Action doesn't affect fluent
    ; TrueConds = [] ->
        build_or(FalseConds, OrFalse),
        R = and(F, neg(OrFalse))
    ; FalseConds = [] ->
        build_or(TrueConds, OrTrue),
        R = or(OrTrue, F)
    ;
        build_or(TrueConds, OrTrue),
        build_or(FalseConds, OrFalse),
        R = or(OrTrue, and(F, neg(OrFalse)))
    ).

/**
 * regress_equality(+Fluent, +Value, +Action, -Result)
 * 
 * Regress F = V through action A.
 */
regress_equality(F, V, A, R) :-
    findall(C, causes_val(A, F, V, C), Conds_V),
    findall((V2, C), (causes_val(A, F, V2, C), V2 \= V), OtherEffects),
    (Conds_V = [] ->
        (OtherEffects = [] ->
            R = (F = V)
        ;
            build_neg_conds(OtherEffects, NegConds),
            R = and(F = V, NegConds)
        )
    ;
        build_or(Conds_V, OrCond),
        (OtherEffects = [] ->
            R = OrCond
        ;
            build_neg_conds(OtherEffects, NegConds),
            R = or(OrCond, and(F = V, NegConds))
        )
    ).

% Build disjunction from list
build_or([C], C) :- !.
build_or([C|Cs], or(C, Rest)) :- build_or(Cs, Rest).

% Build conjunction of negated conditions
build_neg_conds([(_, C)], neg(C)) :- !.
build_neg_conds([(_, C)|Es], and(neg(C), Rest)) :- build_neg_conds(Es, Rest).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  FORMULA SIMPLIFICATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/**
 * simplify_formula(+Formula, -Simplified)
 * 
 * Apply boolean simplification rules iteratively until fixpoint.
 */
simplify_formula(F, S) :-
    simplify_iter(F, S, 10).

simplify_iter(F, F, 0) :- !.
simplify_iter(F_Old, F_Final, N) :-
    N > 0,
    simplify_step(F_Old, F_New),
    (F_Old == F_New ->
        F_Final = F_New
    ;
        N1 is N - 1,
        simplify_iter(F_New, F_Final, N1)
    ).

% Constants
simplify_step(true, true) :- !.
simplify_step(false, false) :- !.

% Negation of constants
simplify_step(neg(true), false) :- !.
simplify_step(neg(false), true) :- !.

% Double negation
simplify_step(neg(neg(P)), S) :- !, simplify_step(P, S).

% AND simplifications
simplify_step(and(true, P), S) :- !, simplify_step(P, S).
simplify_step(and(P, true), S) :- !, simplify_step(P, S).
simplify_step(and(false, _), false) :- !.
simplify_step(and(_, false), false) :- !.
simplify_step(and(P, P), S) :- !, simplify_step(P, S).
simplify_step(and(P, neg(P)), false) :- !.
simplify_step(and(neg(P), P), false) :- !.

% OR simplifications
simplify_step(or(true, _), true) :- !.
simplify_step(or(_, true), true) :- !.
simplify_step(or(false, P), S) :- !, simplify_step(P, S).
simplify_step(or(P, false), S) :- !, simplify_step(P, S).
simplify_step(or(P, P), S) :- !, simplify_step(P, S).
simplify_step(or(P, neg(P)), true) :- !.
simplify_step(or(neg(P), P), true) :- !.

% Implication simplifications
simplify_step(impl(true, P), S) :- !, simplify_step(P, S).
simplify_step(impl(false, _), true) :- !.
simplify_step(impl(_, true), true) :- !.
simplify_step(impl(P, false), neg(S)) :- !, simplify_step(P, S).
simplify_step(impl(P, P), true) :- !.

% Inequality simplifications
simplify_step(O1 \= O2, false) :- O1 == O2, !.
simplify_step(O1 \= O2, true) :- ground(O1), ground(O2), O1 \== O2, !.

% Recursive simplification
simplify_step(and(P1, P2), R) :- !,
    simplify_step(P1, S1),
    simplify_step(P2, S2),
    combine_and(S1, S2, R).

simplify_step(or(P1, P2), R) :- !,
    simplify_step(P1, S1),
    simplify_step(P2, S2),
    combine_or(S1, S2, R).

simplify_step(neg(P), R) :- !,
    simplify_step(P, S),
    combine_neg(S, R).

simplify_step(impl(P1, P2), R) :- !,
    simplify_step(P1, S1),
    simplify_step(P2, S2),
    combine_impl(S1, S2, R).

simplify_step(some(V, P), some(V, S)) :- !, simplify_step(P, S).
simplify_step(all(V, P), all(V, S)) :- !, simplify_step(P, S).
simplify_step(some((V, D), P), some((V, D), S)) :- !, simplify_step(P, S).
simplify_step(all((V, D), P), all((V, D), S)) :- !, simplify_step(P, S).

% Proc-defined simplification: if proc(P, Val) succeeds with Val=true/false,
% use that value directly instead of calling P
% This handles domain-specific simplifications like:
%   proc(door_open(O), false) :- \+ has_door(O).
%   proc(movable(O), true) :- movable(O).
simplify_step(P, Val) :-
    callable(P),
    proc(P, Val),
    (Val == true ; Val == false), !.

% Base case
simplify_step(P, P).

% Combiner helpers
combine_and(false, _, false) :- !.
combine_and(_, false, false) :- !.
combine_and(true, S, S) :- !.
combine_and(S, true, S) :- !.
combine_and(S1, neg(S1), false) :- !.
combine_and(neg(S1), S1, false) :- !.
combine_and(S1, S2, and(S1, S2)).

combine_or(true, _, true) :- !.
combine_or(_, true, true) :- !.
combine_or(false, S, S) :- !.
combine_or(S, false, S) :- !.
combine_or(S1, neg(S1), true) :- !.
combine_or(neg(S1), S1, true) :- !.
combine_or(S1, S2, or(S1, S2)).

combine_neg(true, false) :- !.
combine_neg(false, true) :- !.
combine_neg(neg(P), P) :- !.
combine_neg(P, neg(P)).

combine_impl(true, S, S) :- !.
combine_impl(false, _, true) :- !.
combine_impl(_, true, true) :- !.
combine_impl(S, false, neg(S)) :- !.
combine_impl(S, S, true) :- !.
combine_impl(S1, S2, impl(S1, S2)).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  PROC EXPANSION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/**
 * max_proc_expansion(+Pattern, -MaxCount)
 * 
 * Define the maximum number of times a proc pattern can be expanded.
 * Pattern is matched using proc_pattern/2 (functor-based matching).
 * 
 * Example:
 *   max_proc_expansion(in(_, _), 6).  % Allow 'in' to expand 6 times (for 6 objects)
 *   max_proc_expansion(empty(_), 1).  % Only expand 'empty' once
 * 
 * Default: 100 (effectively unlimited for most use cases)
 */
default_max_proc_expansion(100).

/**
 * get_max_expansion(+Proc, -Max)
 * 
 * Get the maximum expansion count for a proc.
 * First tries user-defined max_proc_expansion/2, then falls back to default.
 */
get_max_expansion(Proc, Max) :-
    proc_pattern(Proc, Pattern),
    (max_proc_expansion(Pattern, Max) -> true
    ; default_max_proc_expansion(Max)).

/**
 * proc_pattern(+Proc, -Pattern)
 * 
 * Extract the pattern (functor with anonymous vars) from a proc term.
 * Example: proc_pattern(in(bread, microwave), in(_, _))
 */
proc_pattern(Proc, Pattern) :-
    functor(Proc, Functor, Arity),
    functor(Pattern, Functor, Arity).

/**
 * expand_procs(+Formula, -Expanded)
 * 
 * Expand proc definitions in Formula, respecting max_proc_expansion limits.
 * Uses iterative expansion until fixpoint or limits reached.
 */
expand_procs(F, E) :-
    expand_procs_iter(F, [], E, 50).  % Max 50 global iterations for safety

expand_procs_iter(F, _, F, 0) :- !.
expand_procs_iter(F_Old, Counts_Old, F_Final, N) :-
    N > 0,
    expand_procs_step(F_Old, Counts_Old, F_New, Counts_New),
    (F_Old == F_New ->
        F_Final = F_New
    ;
        N1 is N - 1,
        expand_procs_iter(F_New, Counts_New, F_Final, N1)
    ).

/**
 * expand_procs_step(+Formula, +CountsIn, -Expanded, -CountsOut)
 * 
 * Single pass of proc expansion.
 * Counts is a list of (Pattern, Count) pairs tracking expansion counts.
 */
% Logical connectives - traverse
expand_procs_step(and(P1, P2), C0, and(E1, E2), C2) :- !,
    expand_procs_step(P1, C0, E1, C1),
    expand_procs_step(P2, C1, E2, C2).

expand_procs_step(or(P1, P2), C0, or(E1, E2), C2) :- !,
    expand_procs_step(P1, C0, E1, C1),
    expand_procs_step(P2, C1, E2, C2).

expand_procs_step(neg(P), C0, neg(E), C1) :- !,
    expand_procs_step(P, C0, E, C1).

expand_procs_step(impl(P1, P2), C0, impl(E1, E2), C2) :- !,
    expand_procs_step(P1, C0, E1, C1),
    expand_procs_step(P2, C1, E2, C2).

expand_procs_step(some(X, P), C0, some(X, E), C1) :- !,
    expand_procs_step(P, C0, E, C1).

expand_procs_step(all(X, P), C0, all(X, E), C1) :- !,
    expand_procs_step(P, C0, E, C1).

expand_procs_step(some((X, Dom), P), C0, some((X, Dom), E), C1) :- !,
    expand_procs_step(P, C0, E, C1).

expand_procs_step(all((X, Dom), P), C0, all((X, Dom), E), C1) :- !,
    expand_procs_step(P, C0, E, C1).

% Proc expansion with per-pattern counting
expand_procs_step(P, C0, E, C_Final) :-
    proc(P, Body), !,
    proc_pattern(P, Pattern),
    get_max_expansion(P, Max),
    get_count(Pattern, C0, Count),
    (Count < Max ->
        % Expand and increment counter
        Count1 is Count + 1,
        set_count(Pattern, Count1, C0, C1),
        expand_procs_step(Body, C1, E, C_Final)
    ;
        % Limit reached, don't expand
        E = P,
        C_Final = C0
    ).

% Base case: not a proc, keep as is
expand_procs_step(P, C, P, C).

/**
 * get_count(+Pattern, +Counts, -Count)
 * Get current expansion count for a pattern (0 if not found).
 */
get_count(Pattern, Counts, Count) :-
    (member((Pat, C), Counts), Pat =@= Pattern -> Count = C ; Count = 0).

/**
 * set_count(+Pattern, +NewCount, +CountsIn, -CountsOut)
 * Set the expansion count for a pattern.
 */
set_count(Pattern, NewCount, CountsIn, CountsOut) :-
    (select((Pat, _), CountsIn, Rest), Pat =@= Pattern ->
        CountsOut = [(Pattern, NewCount)|Rest]
    ;
        CountsOut = [(Pattern, NewCount)|CountsIn]
    ).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  HELPER PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Note: prim_fluent/1 is defined in eval_bat.pl
% Note: system_action/1 is declared multifile above

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  END OF WP COMPUTATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
