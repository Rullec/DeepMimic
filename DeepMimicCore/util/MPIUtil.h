
#pragma once
class cMPIUtil
{
public:
    static bool IsInited();
    static bool InitMPI();
    static int GetCommSize();
    static int GetWorldRank();
    static void SetBarrier();
    static void GetDoubleData(double *ptr, int count, int source, int tag);
    static void SendDoubleData(double *ptr, int count, int source, int tag);
    static void Finalize();
};