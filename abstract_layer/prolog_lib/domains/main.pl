/**
 * Tabletop Domain - Main Entry Point
 * 
 * This file loads all required modules and provides entry points
 * for running the tabletop domain example.
 * 
 * Usage:
 *   swipl prolog_lib/domains/main.pl
 *   ?- demo.
 * 
 * @author IndiGolog Team
 */

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  LOAD REQUIRED MODULES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load formula handling utilities (quantifier expansion)
:- ensure_loaded('../core/formula').

% Load evaluator (required for holds/2, calc_arg/3)
:- ensure_loaded('../eval/eval_bat').

% Load transformation system
:- ensure_loaded('../interpreters/transfinal').

% Load WP computation module
:- ensure_loaded('../eval/wp_computation').

% Load the domain definition
:- ensure_loaded('../../../meta_model/system_model').

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  CONTROLLER PROCEDURES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Simple heating procedure: put bread on plate, plate in microwave, turn on
proc(heat_bread, [
    put(bread, plate),
    put(plate, microwave),
    close(microwave),
    turn_on(microwave)
]).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  DEMO ENTRY POINTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

demo :-
    writeln('=== Tabletop Domain Demo ==='),
    nl,
    % Initialize the evaluator (loads initially/2 into currently/2)
    initialize(evaluator),
    demo_wp,
    nl,
    demo_holds.

demo_wp :-
    writeln('--- WP Computation Examples ---'),
    nl,
    
    writeln('1. WP([put(bread, plate)], loc(bread, plate)):'),
    (wp([put(bread, plate)], loc(bread, plate), WP1) ->
        format('   ~w~n', [WP1])
    ;
        writeln('   Failed')
    ),
    nl,
    
    writeln('2. WP([put(bread, plate), put(plate, microwave)], loc(plate, microwave)):'),
    (wp([put(bread, plate), put(plate, microwave)], loc(plate, microwave), WP2) ->
        format('   ~w~n', [WP2])
    ;
        writeln('   Failed')
    ),
    nl,
    
    writeln('3. WP([close(microwave), turn_on(microwave)], running(microwave)):'),
    (wp([close(microwave), turn_on(microwave)], running(microwave), WP3) ->
        format('   ~w~n', [WP3])
    ;
        writeln('   Failed')
    ),
    nl.

demo_holds :-
    writeln('--- Holds Evaluation (Initial State) ---'),
    nl,
    
    check_holds(loc(bread, table)),
    check_holds(running(microwave)),
    check_holds(door_open(microwave)),
    nl.

check_holds(F) :-
    (holds(F, []) ->
        format('  ~w = true~n', [F])
    ;
        format('  ~w = false~n', [F])
    ).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  QUICK TEST PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Test WP computation
test_wp :-
    writeln('Testing WP computation...'),
    wp([put(bread, plate)], loc(bread, plate), WP),
    format('WP = ~w~n', [WP]).

% Test with turn_on (complex precondition)
test_turn_on :-
    writeln('Testing turn_on WP...'),
    wp([turn_on(microwave)], running(microwave), WP),
    format('WP = ~w~n', [WP]).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  END OF MAIN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
