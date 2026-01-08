/**
 * Formula Handling Utilities for INDIGOLOG
 * 
 * This module provides quantifier expansion utilities for domains with
 * finite object sets. The WP computation module (wp_computation.pl)
 * handles formula simplification internally.
 * 
 * QUANTIFIER EXPANSION
 * ====================
 * 
 * For finite domains D = {o₁, o₂, ..., oₙ}:
 *   ∃x∈D.P(x) → P(o₁) ∨ P(o₂) ∨ ... ∨ P(oₙ)
 *   ∀x∈D.P(x) → P(o₁) ∧ P(o₂) ∧ ... ∧ P(oₙ)
 * 
 * This is COMPLETE for finite domains because every element is enumerated.
 * 
 * Usage:
 *   :- ensure_loaded('../lib/formula').
 * 
 * @author IndiGolog Team
 */

:- module(formula, [
    some_expanded/3,
    all_expanded/3
]).

:- use_module(utils, [subv/4]).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  QUANTIFIER EXPANSION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/**
 * some_expanded((Var, Domain), Formula, ExpandedFormula)
 * 
 * Expands ∃Var∈Domain.Formula to a disjunction over all objects in Domain.
 * 
 * Example:
 *   some_expanded((x, [a,b,c]), p(x), Expanded)
 *   Results in: or(p(a), or(p(b), p(c)))
 */
some_expanded((Var, Domain), Formula, ExpandedFormula) :-
    get_domain_list(Domain, Objects),
    expand_disjunction(Var, Formula, Objects, ExpandedFormula).

/**
 * all_expanded((Var, Domain), Formula, ExpandedFormula)
 * 
 * Expands ∀Var∈Domain.Formula to a conjunction over all objects in Domain.
 * 
 * Example:
 *   all_expanded((x, [a,b,c]), p(x), Expanded)
 *   Results in: and(p(a), and(p(b), p(c)))
 */
all_expanded((Var, Domain), Formula, ExpandedFormula) :-
    get_domain_list(Domain, Objects),
    expand_conjunction(Var, Formula, Objects, ExpandedFormula).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  HELPER PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get list from domain (handles both list and predicate name)
get_domain_list(D, L) :- 
    is_list(D) -> L = D ; (P =.. [D, L], call(P)).

% Build disjunction: ∃x.P(x) → P(o1) ∨ P(o2) ∨ ...
expand_disjunction(_, _, [], false) :- !.
expand_disjunction(Var, Formula, [Obj], Substituted) :- !,
    subv(Var, Obj, Formula, Substituted).
expand_disjunction(Var, Formula, [Obj|Rest], or(Substituted, RestOr)) :-
    subv(Var, Obj, Formula, Substituted),
    expand_disjunction(Var, Formula, Rest, RestOr).

% Build conjunction: ∀x.P(x) → P(o1) ∧ P(o2) ∧ ...
expand_conjunction(_, _, [], true) :- !.
expand_conjunction(Var, Formula, [Obj], Substituted) :- !,
    subv(Var, Obj, Formula, Substituted).
expand_conjunction(Var, Formula, [Obj|Rest], and(Substituted, RestAnd)) :-
    subv(Var, Obj, Formula, Substituted),
    expand_conjunction(Var, Formula, Rest, RestAnd).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  END OF MODULE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
