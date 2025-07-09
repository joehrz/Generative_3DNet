# Project Context for Claude Code

## Developer Profile
- **Primary Languages**: C++/CUDA, Python
- **Development Environment**: [Your OS - Linux/Windows/macOS]
- **IDE/Editor Preferences**: [VS Code/CLion/Vim/etc.]
- **Coding Style**: Professional, performance-oriented, well-documented

## Tech Stack

### C++/CUDA Development
- **C++ Standard**: C++17/C++20 (specify your preference)
- **CUDA Version**: CUDA 12.x (specify your version)
- **Compiler**: nvcc, g++/clang++
- **Build System**: CMake
- **GPU Architecture**: [e.g., sm_75, sm_80, sm_86]
- **Libraries**: 
  - CUDA Runtime API
  - cuBLAS, cuFFT, cuSPARSE (as needed)
  - Thrust (for parallel algorithms)
  - [Add your specific libraries]

### Python Development
- **Python Version**: Python 3.9+ 
- **Package Manager**: pip/conda
- **Virtual Environment**: venv/conda
- **Key Libraries**:
  - NumPy, SciPy
  - CuPy (for GPU computing)
  - Numba (for JIT compilation)
  - PyTorch/TensorFlow (if applicable)
  - matplotlib, pandas (for data work)
  - pytest (for testing)

## Project Structure

```
project/
├── src/                    # Source code
│   ├── cuda/              # CUDA kernels and device code
│   ├── cpp/               # C++ host code
│   └── python/            # Python modules
├── include/               # Header files
├── tests/                 # Unit and integration tests
│   ├── cpp/              # C++ tests (Google Test/Catch2)
│   └── python/           # Python tests (pytest)
├── scripts/              # Build and utility scripts
├── docs/                 # Documentation
├── examples/             # Example usage
└── CMakeLists.txt        # Build configuration
```

## Build Commands

### C++/CUDA
```bash
# Configure build
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build project
cmake --build build --parallel

# Build with debug info
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DCUDA_SEPARABLE_COMPILATION=ON

# Clean build
rm -rf build/
```

### Python
```bash
# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Run tests
pytest tests/

# Format code
black src/ tests/
isort src/ tests/
```

## Code Style Guidelines

### C++/CUDA Best Practices

#### Naming Conventions
- **Variables and functions**: `snake_case` or `camelCase` (choose one consistently)
- **Classes and Types**: `PascalCase`
- **Constants and Enums**: `UPPER_CASE` or `kConstantName`
- **Private members**: `member_name_` (trailing underscore)
- **CUDA kernels**: `kernel_name_kernel`
- **Namespaces**: `lowercase` or `snake_case`
- **Files**: `snake_case.cpp`, `PascalCase.hpp`

#### Modern C++ Features (C++17/20/23)
- **Use `auto`**: For type deduction, especially with iterators and complex types
- **Range-based for loops**: `for (const auto& item : container)`
- **Smart pointers**: `std::unique_ptr`, `std::shared_ptr`, avoid raw pointers for ownership
- **Move semantics**: Implement move constructors/assignment for resource-heavy classes
- **constexpr**: Use for compile-time constants and functions when possible
- **std::optional**: For values that may not exist, avoid null pointers
- **std::variant**: Type-safe unions
- **std::string_view**: For read-only string parameters (no ownership)
- **Structured bindings**: `auto [key, value] = map.insert(...)`

#### Memory Management Excellence
```cpp
// RAII - Resource Acquisition Is Initialization
class GpuMemory {
    void* ptr_;
    size_t size_;
public:
    GpuMemory(size_t size) : size_(size) {
        CHECK_CUDA(cudaMalloc(&ptr_, size));
    }
    ~GpuMemory() { cudaFree(ptr_); }
    
    // Delete copy, enable move
    GpuMemory(const GpuMemory&) = delete;
    GpuMemory& operator=(const GpuMemory&) = delete;
    GpuMemory(GpuMemory&& other) noexcept;
    GpuMemory& operator=(GpuMemory&& other) noexcept;
};

// Smart pointer usage
std::unique_ptr<LargeObject> obj = std::make_unique<LargeObject>(args);
std::shared_ptr<SharedResource> resource = std::make_shared<SharedResource>();

// Container memory management
std::vector<Data> vec;
vec.reserve(expected_size);  // Avoid reallocations
```

#### Performance Best Practices
- **Pass by const reference**: `void func(const LargeObject& obj)`
- **Return value optimization**: Rely on RVO/NRVO, avoid unnecessary copies
- **Reserve container capacity**: `vector.reserve()`, `unordered_map.reserve()`
- **Use emplace**: `vec.emplace_back(args)` instead of `push_back(Object(args))`
- **Prefer algorithms**: Use `<algorithm>` functions over hand-written loops
- **Cache-friendly data structures**: Prefer `std::vector` over `std::list`
- **Minimize virtual function calls**: In hot paths
- **Use `noexcept`**: Mark functions that don't throw exceptions

#### Const Correctness
```cpp
class DataProcessor {
    mutable std::mutex mutex_;  // mutable for const member functions
    std::vector<int> data_;
    
public:
    // Const member function
    size_t size() const noexcept { return data_.size(); }
    
    // Const and non-const overloads
    const int& operator[](size_t idx) const { return data_[idx]; }
    int& operator[](size_t idx) { return data_[idx]; }
    
    // Const parameters
    void process(const std::vector<int>& input) const;
};

// Const correctness in algorithms
auto find_element = [](const auto& container, const auto& value) -> const auto* {
    auto it = std::find(container.begin(), container.end(), value);
    return it != container.end() ? &(*it) : nullptr;
};
```

#### Error Handling Patterns
```cpp
// Use exceptions for exceptional cases
class ComputationError : public std::runtime_error {
public:
    ComputationError(const std::string& msg) : std::runtime_error(msg) {}
};

// RAII for exception safety
class CudaContext {
public:
    CudaContext() { CHECK_CUDA(cudaDeviceReset()); }
    ~CudaContext() { cudaDeviceReset(); }  // Always cleanup
};

// Expected pattern for recoverable errors (C++23)
#include <expected>
std::expected<Result, ErrorCode> compute_something(const Input& input) noexcept;

// Error handling with optional
std::optional<r> try_compute(const Input& input) noexcept;
```

#### Template Best Practices
```cpp
// Use concepts (C++20) for template constraints
template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

template<Numeric T>
T add(T a, T b) { return a + b; }

// SFINAE for older standards
template<typename T>
std::enable_if_t<std::is_arithmetic_v<T>, T> add(T a, T b) {
    return a + b;
}

// Perfect forwarding
template<typename T>
void process(T&& value) {
    some_function(std::forward<T>(value));
}

// Template specialization
template<typename T>
struct TypeTraits {
    static constexpr bool is_gpu_compatible = false;
};

template<>
struct TypeTraits<float> {
    static constexpr bool is_gpu_compatible = true;
};
```

#### STL and Algorithm Usage
```cpp
// Prefer algorithms over raw loops
std::transform(input.begin(), input.end(), output.begin(), 
               [](const auto& x) { return process(x); });

// Use parallel algorithms (C++17)
std::transform(std::execution::par_unseq, 
               input.begin(), input.end(), output.begin(),
               [](const auto& x) { return expensive_process(x); });

// Range algorithms (C++20)
auto filtered = input | std::views::filter([](int x) { return x > 0; })
                     | std::views::transform([](int x) { return x * 2; });

// Prefer standard containers
std::vector<T>          // Default choice for sequences
std::array<T, N>        // Fixed-size arrays
std::unordered_map<K,V> // Hash tables
std::set<T>             // Sorted unique elements
std::deque<T>           // Double-ended queue
```

#### CUDA-Specific C++ Practices
```cpp
// Prefer thrust for parallel algorithms
#include <thrust/transform.h>
#include <thrust/reduce.h>

thrust::transform(thrust::device, input.begin(), input.end(), 
                  output.begin(), functor());

// Use __restrict__ for performance
__global__ void kernel(const float* __restrict__ input,
                       float* __restrict__ output,
                       size_t n);

// Template kernels for type safety
template<typename T>
__global__ void process_kernel(const T* input, T* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = process_element(input[idx]);
    }
}

// CUDA memory management with RAII
class CudaDeviceMemory {
    void* ptr_ = nullptr;
    size_t size_ = 0;
    
public:
    template<typename T>
    explicit CudaDeviceMemory(size_t count) 
        : size_(count * sizeof(T)) {
        CHECK_CUDA(cudaMalloc(&ptr_, size_));
    }
    
    ~CudaDeviceMemory() {
        if (ptr_) cudaFree(ptr_);
    }
    
    template<typename T>
    T* get() const { return static_cast<T*>(ptr_); }
};
```

#### Code Organization
```cpp
// Header guards vs pragma once
#pragma once  // Preferred for modern compilers

// Forward declarations to reduce compilation time
class LargeClass;
std::unique_ptr<LargeClass> create_large_object();

// Pimpl idiom for ABI stability
class PublicInterface {
    class Impl;
    std::unique_ptr<Impl> pimpl_;
public:
    PublicInterface();
    ~PublicInterface();
    // ... public interface
};

// Namespace organization
namespace myproject {
namespace gpu {
    void launch_kernel();
}
namespace cpu {
    void process_data();
}
}  // namespace myproject
```

#### Debugging and Profiling Helpers
```cpp
// Debug-only code
#ifndef NDEBUG
    #define DEBUG_PRINT(x) std::cout << #x << " = " << (x) << std::endl
#else
    #define DEBUG_PRINT(x)
#endif

// Timing utilities
class Timer {
    std::chrono::high_resolution_clock::time_point start_;
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
};

// CUDA timing
class CudaTimer {
    cudaEvent_t start_, stop_;
public:
    CudaTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
        cudaEventRecord(start_);
    }
    
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    
    float elapsed_ms() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        float ms;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }
};
```

#### Compilation Best Practices
- **Use precompiled headers**: For large projects with stable dependencies
- **Enable all warnings**: `-Wall -Wextra -Wpedantic`
- **Treat warnings as errors**: `-Werror`
- **Use sanitizers**: `-fsanitize=address,undefined` for debugging
- **Link-time optimization**: `-flto` for release builds
- **Debug symbols**: `-g` for debugging, `-g0` for release

#### Code Safety and Security
```cpp
// Avoid buffer overflows
std::array<char, 256> buffer{};  // Zero-initialized
std::string safe_string(data, length);  // Length-checked

// Use span for array parameters (C++20)
void process_data(std::span<const float> data);

// Avoid signed/unsigned comparisons
for (size_t i = 0; i < container.size(); ++i) { /* ... */ }

// Secure memory clearing
void secure_memset(void* ptr, size_t size) {
    volatile char* vptr = static_cast<volatile char*>(ptr);
    for (size_t i = 0; i < size; ++i) {
        vptr[i] = 0;
    }
}
```

### Python
- **Style**: Follow PEP 8
- **Type Hints**: Use type annotations for function signatures
- **Documentation**: Use Google-style docstrings
- **Imports**: Group standard library, third-party, and local imports
- **GPU Code**: Use CuPy for GPU arrays, Numba for JIT compilation

## Error Handling

### C++/CUDA
```cpp
// Always check CUDA errors
#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)
```

### Python
- Use try-except blocks for GPU operations
- Handle CuPy memory errors gracefully
- Validate array shapes and data types

## Testing Guidelines

### C++/CUDA
- Use Google Test or Catch2 for unit tests
- Test both CPU and GPU code paths
- Include performance benchmarks for critical kernels
- Test with different input sizes and edge cases

### Python  
- Use pytest for all tests
- Test GPU and CPU implementations separately
- Include property-based tests with Hypothesis when appropriate
- Mock expensive GPU operations in unit tests

## Development Workflow

1. **Feature Development**:
   - Create feature branch from main
   - Write tests first (TDD approach)
   - Implement feature with proper error handling
   - Profile performance-critical code
   - Update documentation

2. **Code Review**:
   - Ensure all tests pass
   - Check memory usage and potential leaks
   - Verify CUDA kernel launch parameters
   - Review for coding standard compliance

3. **Git Workflow**:
   - Use conventional commits (feat:, fix:, docs:, etc.)
   - Squash related commits before merging
   - Use descriptive commit messages

## Performance Considerations

### CUDA Optimization
- **Memory Access**: Ensure coalesced global memory access
- **Occupancy**: Use CUDA Occupancy Calculator for optimal block size
- **Shared Memory**: Use shared memory for data reuse
- **Streams**: Use CUDA streams for overlapping computation and memory transfer

### Python/GPU Integration
- **Memory Management**: Minimize CPU-GPU transfers
- **Batch Processing**: Process data in appropriate batch sizes
- **Profiling**: Use CuPy profiler and nvprof for performance analysis

## Dependencies and Environment

### Required Tools
- NVIDIA GPU with compute capability ≥ 6.0
- CUDA Toolkit (version specified above)
- CMake 3.18+
- GCC/Clang with C++17 support
- Python development environment

### Environment Variables
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## Common Patterns

### CUDA Kernel Launch
```cpp
// Always specify block and grid dimensions explicitly
dim3 blockSize(256);
dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
kernel_name<<<gridSize, blockSize>>>(args);
CHECK_CUDA(cudaGetLastError());
```

### Python-CUDA Integration
```python
import cupy as cp
import numpy as np

# Prefer CuPy arrays for GPU operations
gpu_array = cp.asarray(cpu_array)
result = cp.some_operation(gpu_array)
cpu_result = cp.asnumpy(result)  # Transfer back to CPU only when needed
```

## Debugging and Profiling

### C++/CUDA
- Use `cuda-gdb` for CUDA debugging
- Enable debug symbols: `-g -G` flags
- Use `nvprof` or Nsight Systems for profiling
- Add printf debugging in kernels sparingly

### Python
- Use `pdb` or IDE debuggers
- CuPy provides memory pool information for debugging leaks
- Use `%timeit` in Jupyter for quick performance checks

## Problem-Solving Methodology

### Extended Thinking and Planning
- **Use Think Commands**: When facing complex problems, use "think", "think hard", "think harder", or "ultrathink" to trigger extended reasoning
- **Plan Before Code**: Always create a detailed plan before implementing. Use scratchpad files (SCRATCHPAD.md) to outline approach
- **Break Down Problems**: Decompose complex features into smaller, manageable components
- **Consider Multiple Approaches**: Evaluate different algorithmic and architectural solutions before choosing

### Implementation Strategy
- **Incremental Development**: Implement features in small, testable increments
- **Test-Driven Development**: Write tests first, then implement to pass them
- **Verify Each Step**: Test every component thoroughly before moving to the next
- **Commit Frequently**: Make atomic commits for each working increment

### Issue Resolution Process
1. **Understand the Problem**: Read error messages, logs, and reproduce the issue
2. **Research Context**: Look at related code, documentation, and similar past issues
3. **Hypothesize Causes**: List potential root causes based on symptoms
4. **Test Hypotheses**: Systematically test each hypothesis with minimal changes
5. **Implement Solution**: Apply the fix with proper error handling
6. **Verify Fix**: Ensure the solution works and doesn't break existing functionality
7. **Document**: Update comments, docs, and CLAUDE.md to prevent recurrence

### Novel Idea Generation
- **Cross-Domain Thinking**: Apply concepts from other fields (algorithms, mathematics, physics)
- **Performance Innovation**: Look for unconventional optimization opportunities
- **API Design**: Consider usability and extensibility when designing interfaces
- **Alternative Implementations**: Explore different algorithmic approaches for the same problem

### Plan Mode Workflow
When starting any significant feature or fix:

1. **Define Success Criteria**: What does "done" look like?
2. **Create Implementation Plan**: List all necessary steps in SCRATCHPAD.md
3. **Identify Dependencies**: Note what needs to be built first
4. **Plan Testing Strategy**: How will you verify each component?
5. **Consider Edge Cases**: What could go wrong?
6. **Estimate Complexity**: Is this a 1-hour or 1-day task?

### Incremental Implementation Pattern
```
Phase 1: Core functionality (minimal viable feature)
├── Write basic tests
├── Implement core logic
├── Test and verify
└── Commit working version

Phase 2: Error handling and edge cases
├── Add comprehensive error checking
├── Handle boundary conditions
├── Test failure scenarios
└── Commit robust version

Phase 3: Optimization and polish
├── Profile performance
├── Optimize critical paths
├── Add documentation
└── Final testing and commit
```

## Main Project Goals Tracking

### Primary Objectives
[Define the core mission of your project here - what problem does it solve?]

### Success Metrics
- **Performance Targets**: [e.g., "Process 1M elements in <100ms"]
- **Accuracy Requirements**: [e.g., "Numerical precision within 1e-6"]
- **Usability Goals**: [e.g., "Simple Python API requiring <5 lines of code"]
- **Scalability Targets**: [e.g., "Handle datasets up to 10GB"]

### Feature Priorities
1. **Critical Path Features**: Must-have functionality for MVP
2. **Important Enhancements**: Significant value-add features
3. **Nice-to-Have**: Polish and convenience features

### Current Sprint Goals
[Update this section regularly with current objectives]
- [ ] Current feature being developed
- [ ] Known issues to resolve
- [ ] Next features in queue
- [ ] Technical debt to address

## Cognitive Strategies for Complex Problems

### When Stuck on a Problem
1. **Step Back**: Explain the problem to yourself in simple terms
2. **Rubber Duck Debug**: Describe your approach step-by-step
3. **Change Perspective**: Approach from a different angle (performance, simplicity, robustness)
4. **Research Similar Problems**: Look for analogous solutions in literature or other codebases
5. **Break It Down Further**: Maybe the problem is still too big

### Design Decision Framework
When choosing between approaches, consider:
- **Performance**: Will this scale to production workloads?
- **Maintainability**: Can future developers easily understand and modify this?
- **Robustness**: How does this handle edge cases and errors?
- **Simplicity**: Is this the simplest solution that works?
- **Extensibility**: Will this support future requirements?

### Innovation Triggers
- **"What if we..."**: Explore unconventional approaches
- **"Why not..."**: Question assumptions and constraints
- **"How would X solve this?"**: Apply domain knowledge from other fields
- **"Can we eliminate..."**: Look for unnecessary complexity
- **"What's the opposite approach?"**: Consider inverse solutions

## Codebase Documentation and Analysis

### Master Documentation Creation
When asked to analyze or document the codebase, create comprehensive documentation that includes:

#### Architecture Overview
- **System Design**: High-level architecture diagram in text/ASCII
- **Component Relationships**: How major modules interact
- **Data Flow**: How information moves through the system
- **Dependencies**: External libraries and internal module dependencies
- **Design Patterns**: Architectural patterns used (MVC, Observer, Factory, etc.)

#### Core Components Analysis
For each major component, document:
```markdown
## Component: [ComponentName]

### Purpose
Clear explanation of what this component does and why it exists

### Key Classes/Functions
- `ClassName`: Brief description and primary responsibility
- `function_name()`: What it does, parameters, return value, side effects

### Data Structures
- Important data types and their relationships
- Memory layout considerations (especially for CUDA code)
- Thread safety characteristics

### Algorithms
- Key algorithms implemented
- Time/space complexity analysis
- Performance characteristics and bottlenecks

### Integration Points
- How this component connects to others
- Public interfaces and APIs
- Event handling or callback patterns
```

#### New Feature Documentation Template
When implementing new features, create documentation following this structure:

```markdown
# Feature: [FeatureName]

## Overview
- **Problem Statement**: What problem does this solve?
- **Solution Approach**: High-level strategy
- **Success Criteria**: How do we know it works?

## Technical Design

### API Design
```cpp
// Public interface with detailed comments
class NewFeature {
public:
    /// Brief description of what this method does
    /// @param input Description of input parameter
    /// @return Description of return value
    /// @throws Exceptions that might be thrown
    Result process(const Input& input) const;
};
```

### Implementation Details
- **Core Algorithm**: Step-by-step explanation
- **Data Structures**: New types introduced
- **Performance Considerations**: Memory usage, computational complexity
- **Error Handling**: How failures are managed
- **Thread Safety**: Concurrency considerations

### CUDA-Specific Details (if applicable)
- **Kernel Design**: Block/grid dimensions, shared memory usage
- **Memory Patterns**: Coalescing, bank conflicts, occupancy
- **Launch Parameters**: How to choose optimal configuration

### Integration
- **Modified Components**: What existing code changed
- **New Dependencies**: Libraries or modules added
- **Backward Compatibility**: Impact on existing APIs

### Testing Strategy
- **Unit Tests**: What specific functionality is tested
- **Integration Tests**: How it works with other components
- **Performance Tests**: Benchmarks and expected results
- **Edge Cases**: Unusual inputs or conditions tested

### Usage Examples
```cpp
// Simple usage example
NewFeature feature;
auto result = feature.process(input_data);

// Advanced usage with error handling
try {
    auto result = feature.process(complex_input);
    if (result.is_valid()) {
        // Use result
    }
} catch (const FeatureException& e) {
    // Handle error
}
```
```

#### Code Explanation Standards
When explaining complex code, use this approach:

1. **Context First**: Explain why this code exists
2. **High-Level Flow**: Describe the overall algorithm
3. **Step-by-Step Breakdown**: Go through complex logic line by line
4. **Performance Notes**: Explain optimization choices
5. **Gotchas**: Point out non-obvious behavior or potential pitfalls

##### Example Code Explanation Format:
```markdown
### Function: `complex_algorithm()`

**Purpose**: Optimizes data layout for GPU processing while maintaining cache efficiency

**Algorithm Overview**:
1. Analyze input data patterns
2. Determine optimal memory layout
3. Perform in-place reorganization
4. Validate result integrity

**Detailed Explanation**:
```cpp
template<typename T>
void complex_algorithm(std::vector<T>& data, const Config& config) {
    // Step 1: Calculate optimal block size based on GPU architecture
    // This uses the device's shared memory size and warp size to determine
    // the most efficient processing units
    const size_t block_size = calculate_optimal_block_size<T>(config.device_props);
    
    // Step 2: Pre-sort data to improve memory coalescing
    // Sorting by access pattern reduces global memory latency by ~40%
    std::sort(data.begin(), data.end(), AccessPatternComparator{});
    
    // ... continue with detailed explanations
}
```

**Performance Characteristics**:
- Time Complexity: O(n log n) due to sorting step
- Space Complexity: O(1) additional memory
- GPU Occupancy: ~75% with optimal block size
- Memory Bandwidth: Achieves ~80% of theoretical peak

**Usage Notes**:
- Input data should be pre-validated
- Config must specify valid device properties
- Not thread-safe; use separate instances for concurrent access
```

#### Troubleshooting Documentation
Create debugging guides for complex issues:

```markdown
# Troubleshooting Guide

## Common Issues

### Issue: GPU Memory Access Errors
**Symptoms**: CUDA runtime errors, segmentation faults in kernels
**Likely Causes**:
1. Out-of-bounds array access
2. Uninitialized device memory
3. Race conditions in kernel code

**Debugging Steps**:
1. Enable CUDA debug mode: `export CUDA_LAUNCH_BLOCKING=1`
2. Use cuda-memcheck: `cuda-memcheck ./your_program`
3. Add bounds checking in debug builds
4. Verify memory allocation sizes

### Issue: Performance Degradation
**Symptoms**: Slower than expected execution times
**Investigation Approach**:
1. Profile with nvprof: `nvprof --metrics all ./program`
2. Check memory bandwidth utilization
3. Analyze occupancy with CUDA profiler
4. Look for serialization bottlenecks
```

#### Documentation Maintenance
- **Update Triggers**: When to regenerate documentation
  - New major features added
  - Significant refactoring completed
  - Performance optimizations implemented
  - API changes made
  
- **Review Process**: 
  - Check accuracy after implementation
  - Verify examples still compile and run
  - Update performance numbers with latest benchmarks
  - Remove obsolete sections

### Instructions for Claude

#### When Creating Master Documentation:
1. **Read All Source Files**: Analyze the entire codebase systematically
2. **Identify Patterns**: Look for recurring design patterns and idioms
3. **Trace Data Flow**: Follow how data moves through the system
4. **Explain Design Decisions**: Infer and document the reasoning behind architectural choices
5. **Highlight Complexity**: Focus extra attention on the most complex or critical components
6. **Use Clear Language**: Explain technical concepts clearly but don't oversimplify
7. **Provide Examples**: Include code snippets to illustrate concepts
8. **Note Performance**: Document performance-critical sections and optimizations

#### Documentation Quality Standards:
- **Accuracy**: All code examples must be syntactically correct
- **Completeness**: Cover all major components and their interactions
- **Clarity**: Use precise technical language but explain complex concepts
- **Actionability**: Include enough detail for implementation and debugging
- **Maintainability**: Structure for easy updates as code evolves

#### Special Focus Areas:
- **Memory Management**: Especially important for CUDA code
- **Performance Hotpaths**: Critical algorithms and optimizations
- **Error Handling**: How failures propagate and are managed
- **Concurrency**: Thread safety and parallel processing patterns
- **API Boundaries**: Public interfaces and their contracts

## Notes for Claude

### Development Approach
- **Think First**: Always use extended thinking for complex problems
- **Plan Thoroughly**: Create detailed implementation plans in scratchpad files
- **Implement Incrementally**: Build in small, testable steps
- **Test Everything**: Verify each component before proceeding
- **Stay Goal-Focused**: Keep the main project objectives in mind
- **Document Decisions**: Explain the reasoning behind design choices

### Code Quality Standards
- **Performance Priority**: Always consider performance implications in suggestions
- **Memory Safety**: Emphasize proper memory management in CUDA code
- **Error Checking**: Include appropriate error checking in all code samples
- **Documentation**: Generate comprehensive docstrings and comments
- **Testing**: Suggest appropriate tests for any new functionality
- **Cross-Platform**: Consider compatibility across different GPU architectures
- **Best Practices**: Follow established CUDA and Python best practices

### Problem-Solving Approach
- **Be Systematic**: Follow the issue resolution process for debugging
- **Think Creatively**: Suggest novel approaches when appropriate
- **Consider Alternatives**: Present multiple solutions when possible
- **Explain Reasoning**: Share the thought process behind recommendations
- **Anticipate Issues**: Point out potential problems before they occur
- **Learn from Mistakes**: Update CLAUDE.md when patterns emerge

## Project-Specific Instructions

[Add any project-specific guidelines, conventions, or requirements here]

### Current Codebase Status
[Update this section to reflect the current state]
- **Last Documentation Update**: [Date]
- **Major Components**: [List key modules/classes]
- **Recent Changes**: [Significant modifications]
- **Known Issues**: [Current bugs or limitations]
- **Performance Bottlenecks**: [Areas needing optimization]

---

*This file should be updated regularly as the project evolves and new patterns emerge.*