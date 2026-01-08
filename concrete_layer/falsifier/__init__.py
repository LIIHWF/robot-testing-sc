"""
Falsifier module for configuration instantiation, testing, and STL-based falsification.

This module provides tools for:
- Instantiating concrete configurations from abstract specifications
- Sampling from parameter domains
- Generating test configurations for robotic tasks
- Executing simulations with generated configurations
- Formal parsing of task expressions using PLY
- STL-based falsification using rtamt and nevergrad
"""

from .instantiate import (
    ConfigurationInstantiator,
    ObjectModelDomain,
    PositionDomain,
    FixtureDomain,
    AbstractConfigParser,
)

from .executor import (
    ConfigurationExecutor,
    BatchExecutor,
    ExecutionConfig,
    DynamicEnvironmentFactory,
    ConditionEvaluator,
)

from .task_parser import (
    TaskParser,
    TaskEvaluator,
    LocCondition,
    Action,
    ConditionalTask,
    SequentialTask,
    parse_and_evaluate,
)

from .falsify import (
    # STL Types
    AtomicPredicate,
    STLFormula,
    TemporalOperator,
    # Robustness Functions
    RobustnessExtractor,
    # Task Converters
    TaskSpecConverter,
    PnPTaskSpecConverter,
    MicrowavePnPTaskSpecConverter,
    FixtureTaskSpecConverter,
    get_task_converter,
    # Monitoring
    STLMonitor,
    # Falsification
    STLFalsifier,
    # CT-based Falsification
    ParameterRange,
    CTParameterSpace,
    CTConfigurationLoader,
    CTBasedFalsifier,
    falsify_ct_configuration,
)

__all__ = [
    # Instantiation
    "ConfigurationInstantiator",
    "ObjectModelDomain",
    "PositionDomain",
    "FixtureDomain",
    "AbstractConfigParser",
    # Execution
    "ConfigurationExecutor",
    "BatchExecutor",
    "ExecutionConfig",
    "DynamicEnvironmentFactory",
    "ConditionEvaluator",
    # Task Parser (PLY-based)
    "TaskParser",
    "TaskEvaluator",
    "LocCondition",
    "Action",
    "ConditionalTask",
    "SequentialTask",
    "parse_and_evaluate",
    # STL Types
    "AtomicPredicate",
    "STLFormula",
    "TemporalOperator",
    # Robustness Functions
    "RobustnessExtractor",
    # Task Converters
    "TaskSpecConverter",
    "PnPTaskSpecConverter",
    "MicrowavePnPTaskSpecConverter",
    "FixtureTaskSpecConverter",
    "get_task_converter",
    # Monitoring
    "STLMonitor",
    # Falsification
    "STLFalsifier",
    # CT-based Falsification
    "ParameterRange",
    "CTParameterSpace",
    "CTConfigurationLoader",
    "CTBasedFalsifier",
    "falsify_ct_configuration",
]
