import json
import sys
import os

# Add the py_lib directory to the path for direct script execution
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'abstract_layer', 'py_lib')
sys.path.insert(0, lib_path)
# Import using core as package name (since py_lib is added to path)
from core.system_model import SystemModel
import itertools

'''
Model: 

# All objects: Bread, Fruit, Vegetable, Plate, Microwave, Drawer, Cabinet, Basket, Table
Bread_Location: enum(na, table, plate, microwave, basket)
Fruit_Location: enum(na, table, plate, microwave, basket)
Vegetable_Location: enum(na, table, plate, drawer, basket, cabinet)
Plate_Location: enum(na, table, microwave, drawer, cabinet)

Drawer_Door_State: enum(na, open, closed)
Cabinet_Door_State: enum(na, open, closed)
Microwave_Door_State: enum(na, open, closed)

'''

def _build_at_most_one(proposition_list):
    # Return a list of individual constraints instead of combining them
    # ACTS constraint syntax: try using ! and && operators with quoted values
    constraints = []
    for proposition_a, proposition_b in itertools.combinations(proposition_list, 2):
        # Try format: !(condition1 && condition2) with quoted values
        constraints.append(f'!({proposition_a} && {proposition_b})')
    return constraints

if __name__ == "__main__":
    system_model = SystemModel("tabletop")
    
    location_domain = {
        'bread': ['na', 'table', 'plate', 'basket', 'drawer', 'cabinet', 'microwave'],
        'fruit': ['na', 'table', 'plate', 'basket', 'drawer', 'cabinet', 'microwave'],
        'vegetable': ['na', 'table', 'plate', 'basket', 'drawer', 'cabinet', 'microwave'],
        'plate': ['na', 'table', 'microwave', 'drawer', 'cabinet', 'basket'],
        'basket': ['na', 'table'],
    }
    
    for object, locations in location_domain.items():
        system_model.add_parameter(f"{object}_Location", "enum", locations)
    
    door_state_domain = ['na', 'open', 'closed']
    system_model.add_parameter("Drawer_Door_State", "enum", door_state_domain)
    system_model.add_parameter("Cabinet_Door_State", "enum", door_state_domain)
    system_model.add_parameter("Microwave_Door_State", "enum", door_state_domain)
    
    system_model.add_parameter("Microwave_Running_State", "enum", ['na', 'stopped', 'running'])
    
    # Microwave_Running_State cannot be running
    system_model.add_constraint("Microwave_Running_State!=\"running\"")
    
    # at most one object in each location (excluding "na" since multiple objects can be na)
    # Exception: "table" allows multiple objects, while other containers (microwave, cabinet, drawer, basket) allow only one
    all_locations = set(itertools.chain(*location_domain.values()))
    # Containers that allow only one object (excluding table)
    single_object_containers = {'microwave', 'cabinet', 'drawer', 'basket', 'plate'}
    
    for location in all_locations:
        # Skip "na" location - multiple objects can be na simultaneously
        if location == 'na':
            continue
        # Skip "table" location - allow multiple objects on table
        if location == 'table':
            continue
        # Only include objects that can actually be placed in this location
        proposition_list = [f"{object}_Location=\"{location}\"" 
                           for object, locations in location_domain.items() 
                           if location in locations]
        # Only add constraint if there are at least 2 objects that can be in this location
        if len(proposition_list) >= 2:
            # Add each constraint separately
            for constraint in _build_at_most_one(proposition_list):
                system_model.add_constraint(constraint)
    
    # for drawer, microwave and cabinet, at most one of them can present, that is least one of them is na
    system_model.add_constraint("!(Microwave_Door_State!=\"na\" && Cabinet_Door_State!=\"na\")")
    system_model.add_constraint("!(Drawer_Door_State!=\"na\" && Microwave_Door_State!=\"na\")")
    system_model.add_constraint("!(Drawer_Door_State!=\"na\" && Cabinet_Door_State!=\"na\")")
    
    # if one is na, then the other's loc cannot be that one
    for location in all_locations:
        if location == 'na':
            continue
        for object, locations in location_domain.items():
            if location in locations and location in location_domain.keys():
                system_model.add_constraint(f"({location}_Location!=\"na\" || {object}_Location!=\"{location}\")")
    
    import argparse

    parser = argparse.ArgumentParser(description="Save and solve tabletop model")
    parser.add_argument('--output', '-o', type=str, default="cache/tabletop_model.txt",
                        help="Output file path (default: cache/tabletop_model.txt)")
    args = parser.parse_args()

    system_model.save(args.output)
    # system_model.solve(args.model_path, args.result_path, args.algorithm)