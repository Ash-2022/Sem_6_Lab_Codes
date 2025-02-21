#ifndef SYMBOL_TABLE_H
#define SYMBOL_TABLE_H

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

#define TableLength 30

int numEntries = 0;

enum identifierType
{
    CHAR = 0,
    INT,
    FLOAT,
    DOUBLE,
    FUNC,
    RANDOM,
    STRING,
};

const char *datatypeName[] = {
    "char", "int", "float", "double", "func" , "unknown" , "string"};
const int datatypeSize[] = {
    sizeof(char), sizeof(int), sizeof(float), sizeof(double) , 0 , -1};

struct token
{
    char *lexeme;
    int index;
    unsigned int rowno, colno; // row number, column number.
    enum identifierType type;
};

typedef struct ListElement
{
    struct token tok;
} symbolTableEntry;

symbolTableEntry TABLE[TableLength];

void initSymbolTable()
{
    for (int i = 0; i < TableLength; i++)
    {
        TABLE[i].tok.colno = 0;
        TABLE[i].tok.rowno = 0;
        TABLE[i].tok.index = i;
        TABLE[i].tok.type = CHAR;
        TABLE[i].tok.lexeme = "";
    }
}

void Display()
{
    int id = 0;
    if (!numEntries)
    {
        printf("No elements in the Symbol Table\n");
        return;
    }
    printf("\t Symbol Table \n");
    printf("Id\tLexName\t\tType\tSize\n");
    do
    {
        printf("%d\t%s\t%s\t%d\n", id, TABLE[id].tok.lexeme,
               datatypeName[TABLE[id].tok.type],
               TABLE[id].tok.type != STRING
                   ? datatypeSize[TABLE[id].tok.type]
                   : strlen(TABLE[id].tok.lexeme) * sizeof(char));
        id++;
    } while (id < numEntries);
}

int HASH(char *str)
{
    // Develop an OpenHash function on a string.
    return numEntries % TableLength;
}

int SEARCH(char *str)
{
    // Write a search routine to check whether a lexeme exists in the Symbol table.
    // Returns 1, if lexeme is found
    // else returns 0
    int val = HASH(str);
    if (TABLE[val].tok.lexeme == str)
        return 1;
    else
        return 0;
}

void INSERT(struct token tk)
{
    int index = HASH(tk.lexeme);
    if (!SEARCH(tk.lexeme))
    {
        // Allocate memory for the lexeme and copy it
        TABLE[index].tok.lexeme = (char *)malloc(strlen(tk.lexeme) + 1);
        if (TABLE[index].tok.lexeme != NULL)
        {
            strcpy(TABLE[index].tok.lexeme, tk.lexeme);
            TABLE[index].tok.rowno = tk.rowno;
            TABLE[index].tok.colno = tk.colno;
            TABLE[index].tok.type = tk.type;
            TABLE[index].tok.index = index;
            numEntries++;
        }
        else
        {
            fprintf(stderr, "Memory allocation failed for lexeme\n");
        }
    }
}

#endif