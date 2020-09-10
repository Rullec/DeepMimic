#include "MPIUtil.h"
#ifdef __APPLE__
#include <mpi.h>
#else
#include <mpi/mpi.h>
#endif

bool cMPIUtil::IsInited()
{
    int is_mpi_init;
    MPI_Initialized(&is_mpi_init);
    return static_cast<bool>(is_mpi_init);
}
bool cMPIUtil::InitMPI()
{
    int is_mpi_init;
    MPI_Initialized(&is_mpi_init);
    if (0 == is_mpi_init)
        MPI_Init(NULL, NULL);
    return true;
}
int cMPIUtil::GetCommSize()
{
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
}
int cMPIUtil::GetWorldRank()
{
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    return world_rank;
}

void cMPIUtil::SetBarrier() { MPI_Barrier(MPI_COMM_WORLD); }

void cMPIUtil::Finalize() { MPI_Finalize(); }

void cMPIUtil::GetDoubleData(double *ptr, int count, int source, int tag)
{
    MPI_Status status;
    MPI_Recv(ptr, count, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);
}

void cMPIUtil::SendDoubleData(double *ptr, int count, int dest, int tag)
{

    MPI_Send(ptr, count, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
}