#include <mpi.h>
#include <stdio.h>

double pow(double num , int pow){
    double result = 1;
    for(int i = 0; i < pow; i++){
        result *= num;
    }
    return result;
}

int main(int argc , char*argv[]){
    int rank , size;
    MPI_Init(&argc , &argv);
    MPI_Comm_size(MPI_COMM_WORLD , &size);
    MPI_Comm_rank(MPI_COMM_WORLD , &rank);
    int m;
    if(!rank){
        printf("Enter Val of M : ");
        fflush(stdout);
        scanf("%d" , &m);
    }
    MPI_Bcast(&m , 1 , MPI_INT , 0 , MPI_COMM_WORLD);
    int totalSize = m * size;
    double input[totalSize];
    if(!rank){
        for(int i = 0; i < totalSize; i++){
            printf("Enter Element %d : " , i);
            fflush(stdout);
            scanf("%lf" , &input[i]);
        }
    }
    double workingBuff[m];
    MPI_Scatter(&input , m , MPI_DOUBLE , &workingBuff , m , MPI_DOUBLE , 0 ,  MPI_COMM_WORLD);
    for(int i = 0; i < m;i++){
        workingBuff[i] = pow(workingBuff[i] , rank+2);
    }
    MPI_Gather(&workingBuff , m , MPI_DOUBLE , &input , m , MPI_DOUBLE , 0 , MPI_COMM_WORLD);
    if(!rank){
        for(int i = 0; i < totalSize; i++){
            printf("%lf " , input[i]);
            fflush(stdout);
        }
        printf("\n");
    }
    MPI_Finalize();
    return 0;
}
