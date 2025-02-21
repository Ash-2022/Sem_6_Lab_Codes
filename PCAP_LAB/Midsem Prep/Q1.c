/*
Take a string input in root
Of length N 
Use standard send to send the string to every other process

In each process
Extract substring of length N/size

If the rank of the process is a prime number
Toggle the vowels in the extracted substring

Use collective communication function to aggregate the modified substrings back in the root
*/

#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

#define MAX_N 100

int isPrime(int x) {
    if (x < 2) return 0;
    if (x == 2) return 1;
    for (int i = 2; i < x; i ++) { 
        if (x % i == 0) return 0;
    }
    return 1;
}


void toggleVowels(char *str, int len) {
    for (int i = 0; i < len; i++) {
        if (strchr("aeiou", str[i])) 
            str[i] = toupper(str[i]);
        else if (strchr("AEIOU", str[i])) 
            str[i] = tolower(str[i]);
    }
}

int main(int argc, char *argv[]) {
    int rank, size, n;
    char input[MAX_N];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        printf("Enter N: ");
        fflush(stdout);
        scanf("%d", &n);

        printf("Enter string of length %d: ", n);
        fflush(stdout);
        scanf("%s", input);

        for (int i = 1; i < size; i++) {
            MPI_Send(&n, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(input, n, MPI_CHAR, i, 1, MPI_COMM_WORLD);
        }
    }
    else {
        MPI_Recv(&n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(input, n, MPI_CHAR, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    int sublen = n / size;

    char subStr[sublen + 1];
    strncpy(subStr, &input[rank * sublen], sublen);
    subStr[sublen] = '\0';

    printf("Process %d received substring: %s\n", rank, subStr);

    if (isPrime(rank)) {
        printf("Process %d is prime, before toggle: %s\n", rank, subStr);
        toggleVowels(subStr, sublen);
        printf("Process %d is prime, after toggle: %s\n", rank, subStr);
    }

    char gatheredStr[MAX_N];
    MPI_Gather(subStr, sublen, MPI_CHAR, gatheredStr, sublen, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        gatheredStr[n] = '\0';
        printf("Modified Str = %s\n", gatheredStr);
    }

    MPI_Finalize();
    return 0;
}
