# Prolog Library - Prolog 代码库

本目录包含所有 Prolog 代码，按功能分类组织。

## 目录结构

```
prolog_lib/
├── core/                    # 核心库模块
│   ├── common.pl           # 通用库
│   ├── formula.pl          # 公式处理工具（量词展开）
│   ├── utils.pl            # 工具函数
│   └── wscplan.pl          # WSCP 规划器
│
├── eval/                    # 评估工具
│   ├── eval_bat.pl         # BAT 评估器（提供 holds/2, calc_arg/3）
│   └── wp_computation.pl   # WP（最弱前置条件）计算模块
│
├── interpreters/            # 解释器
│   ├── transfinal.pl              # 转换系统（主文件）
│   ├── transfinal-ext.pl          # 扩展构造
│   ├── transfinal-search.pl       # 搜索构造（INDIGOLOG）
│   └── transfinal-congolog.pl     # ConGolog 支持
│
└── domains/                 # 领域定义
    └── main.pl              # Tabletop 领域主入口点
```

## 模块分类说明

### core/ - 核心库模块
- **common.pl**: 通用库，提供基础工具和定义
- **formula.pl**: 公式处理工具，提供量词展开功能（`some_expanded/3`, `all_expanded/3`）
- **utils.pl**: 工具函数，提供变量替换、字符串转换等功能
- **wscplan.pl**: WSCP（Weakly Specified ConGolog Programs）规划器

### eval/ - 评估工具
- **eval_bat.pl**: BAT（Basic Action Theory）评估器，实现 `holds/2` 和 `calc_arg/3` 谓词
- **wp_computation.pl**: WP（最弱前置条件）计算模块，提供 `wp/3` 谓词

### interpreters/ - 解释器
- **transfinal.pl**: 转换系统主文件，定义 `trans/4` 和 `final/2` 谓词
- **transfinal-ext.pl**: 扩展构造支持
- **transfinal-search.pl**: 搜索构造支持（INDIGOLOG），依赖 `core/wscplan.pl`
- **transfinal-congolog.pl**: ConGolog 支持（包括 Golog）

### domains/ - 领域定义
- **main.pl**: Tabletop 领域的主入口点，加载所有必需模块并提供演示功能

**注意**: 领域定义文件 `system_model.pl` 位于 `meta_model/` 目录中。

## 使用示例

### 加载 Tabletop 领域

```prolog
% 从项目根目录
swipl prolog_lib/domains/main.pl

% 或者
swipl
?- ['prolog_lib/domains/main'].
?- demo.
```

### 单独加载模块

```prolog
% 加载公式处理工具
:- ensure_loaded('prolog_lib/core/formula').

% 加载评估器
:- ensure_loaded('prolog_lib/eval/eval_bat').

% 加载转换系统
:- ensure_loaded('prolog_lib/interpreters/transfinal').

% 加载 WP 计算模块
:- ensure_loaded('prolog_lib/eval/wp_computation').
```

## 依赖关系

### 模块依赖图

```
main.pl
├── core/formula.pl
│   └── core/utils.pl
├── eval/eval_bat.pl
├── interpreters/transfinal.pl
│   ├── interpreters/transfinal-ext.pl
│   ├── interpreters/transfinal-search.pl
│   │   └── core/wscplan.pl  (WSCP 规划器)
│   └── interpreters/transfinal-congolog.pl
├── eval/wp_computation.pl
└── meta_model/system_model.pl
```

## 迁移说明

这些 Prolog 文件原本分散在多个目录中：
- `lib/` → `prolog_lib/core/`
- `eval/` → `prolog_lib/eval/`
- `interpreters/` → `prolog_lib/interpreters/`
- `meta_model/domain.pl` → `meta_model/system_model.pl`（重命名并保留在 meta_model）
- `system_model/tabletop/main.pl` → `prolog_lib/domains/main.pl`

### 路径更新

所有文件中的 `ensure_loaded` 路径已更新为新的相对路径：
- `main.pl` 中的路径已更新（引用 `meta_model/system_model.pl`）
- `transfinal-search.pl` 中对 `wscplan.pl` 的引用已更新
- 其他文件中的相对路径引用已保持正确

## 注意事项

- 所有路径引用使用相对路径，确保从项目根目录加载时正常工作
- `formula.pl` 使用 `use_module(utils, ...)` 引用同目录下的 `utils.pl`
- `transfinal.pl` 中的引用都是同一目录下的文件，使用文件名即可
- 建议从项目根目录加载文件，以确保路径解析正确
