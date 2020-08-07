set(IDSOLVER_SRC
    BuildIDSolver.cpp
    DisplayIDSolver.cpp
    IDSolver.cpp
    InteractiveIDSolver.cpp
    OfflineSolveIDSolver.cpp
    OnlineIDSolver.cpp
    SampleIDSolver.cpp
)
add_library(IDSOLVER_LIB ${IDSOLVER_SRC})