/**
 * Tabletop Domain: A Clean Situation Calculus Example
 * 
 * This domain models a tabletop scenario with objects that can be placed
 * in containers. It demonstrates the core situation calculus modeling approach.
 * 
 * STRUCTURE OF A DOMAIN DEFINITION
 * ================================
 * 
 * 1. OBJECTS (object/1)
 *    - Define all objects in the domain
 *    - Example: object(bread), object(plate), object(microwave)
 * 
 * 2. RIGID PREDICATES (Type predicates)
 *    - Properties that never change (not situation-dependent)
 *    - Type predicates: is_item/1, is_container/1, etc.
 *    - Attribute predicates: movable/1, heatable/1, etc.
 * 
 * 3. FLUENTS (rel_fluent/1, fun_fluent/1)
 *    - Properties that can change
 *    - Relational fluents: loc(O1, O2), door_open(O), running(O)
 *    - Functional fluents: temperature(O) (returns a value)
 * 
 * 4. DERIVED FLUENTS (proc/2)
 *    - Fluents defined in terms of other fluents
 *    - Example: in(O1, O2) as transitive closure of loc
 * 
 * 5. ACTIONS (prim_action/1)
 *    - Primitive actions that agents can perform
 *    - Example: put(O1, O2), open(O), close(O), turn_on(O)
 * 
 * 6. SUCCESSOR STATE AXIOMS (causes_true/3, causes_false/3)
 *    - Define how actions change fluents
 *    - causes_true(Action, Fluent, Condition): Action makes Fluent true when Condition holds
 *    - causes_false(Action, Fluent, Condition): Action makes Fluent false when Condition holds
 * 
 * 7. PRECONDITION AXIOMS (poss/2)
 *    - Define when actions are possible
 *    - poss(Action, Condition): Action is possible when Condition holds
 * 
 * 8. INITIAL CONDITIONS (initially/2)
 *    - Define the initial state
 *    - initially(Fluent, Value): Fluent has Value in the initial state
 * 
 * @author IndiGolog Team
 */

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  MODULE DECLARATIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- dynamic controller/1.
:- multifile(proc/2).
:- discontiguous
    fun_fluent/1,
    rel_fluent/1,
    proc/2,
    causes_true/3,
    causes_false/3,
    poss/2,
    initially/2,
    object/1,
    is_item/1,
    is_container/1,
    is_surface/1,
    movable/1,
    heatable/1,
    require_heat/1,
    has_door/1,
    has_running_state/1,
    rigid_predicate/1.

% No caching needed (required because cache/1 is static)
cache(_) :- fail.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  SECTION 1: OBJECTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% === Table (Surface) ===
object(table).
is_surface(table).

% === Bread (Item) ===
object(bread).
is_item(bread).
movable(bread).
heatable(bread).
require_heat(bread).

% === Fruit (Item) ===
object(fruit).
is_item(fruit).
movable(fruit).

% === Vegetable (Item) ===
object(vegetable).
is_item(vegetable).
movable(vegetable).
heatable(vegetable).
require_heat(vegetable).

% === Plate (Container/Surface) ===
object(plate).
is_container(plate).
movable(plate).
heatable(plate).

% === Microwave (Container with door and running state) ===
object(microwave).
is_container(microwave).
has_door(microwave).
has_running_state(microwave).

% === Cabinet (Container with door) ===
object(cabinet).
is_container(cabinet).
has_door(cabinet).

% === Drawer (Container with door) ===
object(drawer).
is_container(drawer).
has_door(drawer).

% === Basket (Container) ===
object(basket).
is_container(basket).

% Object list for quantifier expansion
objects([table, bread, fruit, vegetable, plate, microwave, cabinet, drawer, basket]).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  SECTION 2: PRIMITIVE FLUENTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% loc(O1, O2): Object O1 is directly located on/in O2
rel_fluent(loc(O1, O2)) :- object(O1), object(O2).

% door_open(O): Door of object O is open
rel_fluent(door_open(O)) :- object(O).

% running(O): Object O is running (e.g., microwave is on)
rel_fluent(running(O)) :- object(O).

% ====== Fluent Simplification Rules ======
% When a fluent is meaningless for a specific object, simplify to false.
% These proc rules are evaluated during WP formula simplification.

% An object cannot be located inside itself
proc(loc(O, O), false) :- object(O).

% Only movable objects can have location, and target must be a container
proc(loc(O1, O2), false) :- 
    object(O1), object(O2), 
    (\+ movable(O1); \+ is_container(O2)).

% Only objects with doors can have door_open state
proc(door_open(O), false) :- object(O), \+ has_door(O).

% Only objects with running state can have running state
proc(running(O), false) :- object(O), \+ has_running_state(O).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  SECTION 3: DERIVED FLUENTS (Procedures)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- PROC EXPANSION LIMITS ---
% For non-recursive 'in' (Option B), no limit needed since each 'in' expands once.
% For recursive 'in' (Option A), limit = N^2 where N is object count.
% This accounts for multiple 'in' calls in the formula, each needing N expansions.
max_proc_expansion(in(_, _), Limit) :- 
    objects(Objs), length(Objs, N), 
    Limit is N * N.  % N^2 to handle all 'in' instances

% --- OPTION A: Recursive definition with expansion limit ---
% Uncomment this if you want the recursive version with max_proc_expansion limit.
% NOTE: The expansion count approach may not work well for deeply recursive procs.
%
% proc(in(O1, O2), ExpandedFormula) :-
%     some_expanded((o_mid, objects), and(loc(O1, o_mid), in(o_mid, O2)), SomeExpanded),
%     ExpandedFormula = or(loc(O1, O2), SomeExpanded).

% --- OPTION B: Non-recursive parametric expansion (RECOMMENDED) ---
% This directly builds the transitive closure up to depth N-1 (max path length).
% No recursive proc calls, so no expansion limits needed.
proc(in(O1, O2), ExpandedFormula) :-
    build_in_transitive(O1, O2, ExpandedFormula).

% Build transitive closure non-recursively:
% in(O1, O2) = loc(O1, O2) ∨ ∃M1.(loc(O1,M1) ∧ loc(M1,O2)) ∨ ∃M1,M2.(loc(O1,M1) ∧ loc(M1,M2) ∧ loc(M2,O2)) ∨ ...
%
% NOTE: MaxDepth is limited to 2 to avoid combinatorial explosion.
% With N objects, depth K generates N^K chains. For N=9, depth=8 would generate 43M chains!
% In practice, nesting depth > 2 is rare for tabletop scenarios.
build_in_transitive(O1, O2, Formula) :-
    objects(Objs),
    MaxDepth = 3,  % Limit max path length to avoid exponential blowup (was: N-1)
    build_in_levels(O1, O2, 0, MaxDepth, Objs, Levels),
    list_to_or(Levels, Formula).

% Build formulas for each depth level [0, 1, ..., MaxDepth]
build_in_levels(_, _, Depth, MaxDepth, _, []) :- Depth > MaxDepth, !.
build_in_levels(O1, O2, Depth, MaxDepth, Objs, [Level|Rest]) :-
    Depth =< MaxDepth,
    build_in_at_depth(O1, O2, Depth, Objs, Level),
    NextDepth is Depth + 1,
    build_in_levels(O1, O2, NextDepth, MaxDepth, Objs, Rest).

% Depth 0: direct location
build_in_at_depth(O1, O2, 0, _, loc(O1, O2)) :- !.

% Depth K > 0: chain of K+1 loc predicates through intermediate objects
% ∃M1,...,Mk.(loc(O1,M1) ∧ loc(M1,M2) ∧ ... ∧ loc(Mk,O2))
build_in_at_depth(O1, O2, Depth, Objs, Formula) :-
    Depth > 0,
    % Generate all possible chains of intermediate objects
    findall(ChainFormula,
            (generate_chain(O1, O2, Depth, Objs, ChainFormula)),
            Chains),
    list_to_or(Chains, Formula).

% Generate a single chain: loc(O1, M1) ∧ loc(M1, M2) ∧ ... ∧ loc(Mn, O2)
generate_chain(O1, O2, Depth, Objs, ChainFormula) :-
    length(Mids, Depth),           % Create list of Depth intermediate vars
    select_mids(Mids, Objs),       % Bind each to an object
    build_loc_chain(O1, Mids, O2, ChainFormula).

% Select intermediate objects (allowing repetition, but excluding trivial cycles)
select_mids([], _) :- !.
select_mids([M|Rest], Objs) :-
    member(M, Objs),
    select_mids(Rest, Objs).

% Build loc chain: loc(O1, M1) ∧ loc(M1, M2) ∧ ... ∧ loc(Mn, O2)
build_loc_chain(O1, [], O2, loc(O1, O2)) :- !.
build_loc_chain(O1, [M|Mids], O2, and(loc(O1, M), RestChain)) :-
    build_loc_chain(M, Mids, O2, RestChain).

% Convert list to disjunction
list_to_or([], false) :- !.
list_to_or([F], F) :- !.
list_to_or([F|Rest], or(F, RestOr)) :-
    list_to_or(Rest, RestOr).

% empty(O): No object is directly located on/in O
%   empty(O) ≡ ∀O'.(¬loc(O', O))
proc(empty(O), ExpandedFormula) :-
    all_expanded((o, objects), neg(loc(o, O)), ExpandedFormula).

% all_inside_heatable(O): All objects inside O are heatable
%   all_inside_heatable(O) ≡ ∀O'.(in(O', O) → heatable(O'))
proc(all_inside_heatable(O), ExpandedFormula) :-
    all_expanded((o, objects), impl(in(o, O), heatable(o)), ExpandedFormula).

% some_inside_requires_heat(O): Some object inside O requires heating
%   some_inside_requires_heat(O) ≡ ∃O'.(in(O', O) ∧ require_heat(O'))
proc(some_inside_requires_heat(O), ExpandedFormula) :-
    some_expanded((o, objects), and(in(o, O), require_heat(o)), ExpandedFormula).

% accessible(O): O can be accessed (not inside a closed container)
%   accessible(O) ≡ ∀C.((has_door(C) ∧ in(O, C)) → door_open(C))
%   i.e., for all containers with doors, if O is inside, the door must be open
proc(accessible(O), ExpandedFormula) :-
    all_expanded((c, objects), 
        impl(and(has_door(c), in(O, c)), door_open(c)), 
        ExpandedFormula).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  RIGID PREDICATES HANDLING (Auto-convert to true/false)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define which predicates are rigid (unchanging across all situations)
rigid_predicate(is_item/1).
rigid_predicate(is_container/1).
rigid_predicate(is_surface/1).
rigid_predicate(movable/1).
rigid_predicate(heatable/1).
rigid_predicate(require_heat/1).
rigid_predicate(has_door/1).
rigid_predicate(has_running_state/1).

% Generic proc rule: auto-convert rigid predicates to true/false
proc(P, true) :-
    functor(P, F, A),
    rigid_predicate(F/A),
    call(P), !.

proc(P, false) :-
    functor(P, F, A),
    rigid_predicate(F/A),
    \+ call(P).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  SECTION 4: ACTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

prim_action(put(O1, O2)) :- object(O1), object(O2).
prim_action(open(O)) :- object(O).
prim_action(close(O)) :- object(O).
prim_action(turn_on(O)) :- object(O).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  SECTION 5: SUCCESSOR STATE AXIOMS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% put(O1, O2) effects:
%   - Makes loc(O1, O2) true
%   - Makes loc(O1, O_prev) false for any O_prev ≠ O2
causes_true(put(O1, O2), loc(O1, O2), true) :- 
    object(O1), object(O2).
causes_false(put(O1, O2), loc(O1, O_prev), true) :- 
    object(O1), object(O2), object(O_prev), O_prev \= O2.

% open(O) effects:
%   - Makes door_open(O) true
%   - Makes running(O) false (opening stops the running state)
causes_true(open(O), door_open(O), true) :- object(O).
causes_false(open(O), running(O), true) :- object(O).

% close(O) effects:
%   - Makes door_open(O) false
causes_false(close(O), door_open(O), true) :- object(O).

% turn_on(O) effects:
%   - Makes running(O) true
causes_true(turn_on(O), running(O), true) :- object(O).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  SECTION 6: PRECONDITION AXIOMS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% poss(put(O1, O2), Cond):
%   Can put O1 on/in O2 if:
%   - O1 is movable
%   - O1 ≠ O2
%   - O2 is empty (simplified: only one object can be on/in O2)
%   - If O2 has a door, it must be open
%   - O2 is a valid destination (surface or container)
%   - O1 must be accessible (not inside a closed container)
%   - O2 must be accessible (not inside a closed container)
poss(put(O1, O2), 
    and(accessible(O1),
    and(accessible(O2),
    and(neg(loc(O1, O2)),
    and(movable(O1),
    and(O1 \= O2,
    and(empty(O2),
    and(impl(has_door(O2), door_open(O2)),
        or(is_surface(O2), is_container(O2)))))))))) :- 
    object(O1), object(O2).

% poss(open(O), Cond):
%   Can open O if:
%   - O has a door
%   - The door is not already open
%   - O is not running (can't open while running)
poss(open(O), 
    and(has_door(O), 
    and(neg(door_open(O)), 
        impl(has_running_state(O), neg(running(O)))))) :- 
    object(O).

% poss(close(O), Cond):
%   Can close O if:
%   - O has a door
%   - The door is open
%   - O is not running
poss(close(O), 
    and(has_door(O), 
    and(door_open(O),
        impl(has_running_state(O), neg(running(O)))))) :- 
    object(O).

% poss(turn_on(O), Cond):
%   Can turn on O if:
%   - O has a running state (like microwave)
%   - O is not already running
%   - Door is closed
%   - All inside objects are heatable
%   - At least one inside object requires heating
poss(turn_on(O), 
    and(has_running_state(O),
    and(neg(running(O)),
    and(neg(door_open(O)),
    and(all_inside_heatable(O),
        some_inside_requires_heat(O)))))) :- O = microwave.

% For objects without running state, turn_on is impossible
poss(turn_on(O), false) :- 
    object(O), \+ has_running_state(O).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  SECTION 7: INITIAL CONDITIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initially, nothing is running
initially(running(O), false) :- object(O).

% Initially, no object is located on itself
initially(loc(O, O), false) :- object(O).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  END OF DOMAIN DEFINITION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




