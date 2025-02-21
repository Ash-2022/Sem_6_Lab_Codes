#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_TOKEN_LEN 100
#define MAX_BUFFER_LEN 1000

enum
{
    EXIT_EOF = 2,
    EXIT_TOKEN_CREATED,
    EXIT_TOKEN_FAILED,
};

enum
{
    IDENTIFIER = 0,
    KEYWORD,
    LITERAL,
    NUMBER,
    SEPERATOR,
    RELOP,
    ARITHMETIC,
    UNKNOWN
};

char tokenType[8][20] = {
    "identifier", "keyword", "literal", "number", "seperator", "relop", "arithmetic", "unknown"};

const char *keywords[] = {
    "auto", "break", "case", "char", "const", "continue", "default",
    "do", "double", "else", "enum", "extern", "float", "for", "goto",
    "if", "int", "long", "register", "return", "short", "signed",
    "sizeof", "static", "struct", "switch", "typedef", "union",
    "unsigned", "void", "volatile", "while"};

const int NUM_KEYWORDS = 32;

typedef struct
{
    char *tokenContent;
    int size;
    int tokenType;
} Token;

typedef struct
{
    Token **tokenBuf;
    int count;
} TokenBuffer;

Token *initToken()
{
    Token *token = (Token *)malloc(sizeof(Token));
    token->tokenContent = (char *)malloc(sizeof(char) * MAX_TOKEN_LEN);
    token->size = 0;
    token->tokenType = UNKNOWN;
    return token;
}

void freeToken(Token * tok){
    free(tok->tokenContent);
    free(tok);
}

void printToken(Token *tok)
{
    printf("<%s , %s>\n", tokenType[tok->tokenType], tok->tokenContent);
}

TokenBuffer *initTokenBuffer()
{
    TokenBuffer *tokenBuf = (TokenBuffer *)malloc(sizeof(TokenBuffer));
    tokenBuf->tokenBuf = (Token **)malloc(sizeof(Token *) * MAX_BUFFER_LEN);
    tokenBuf->count = 0;
    return tokenBuf;
}

int isKeyword(char *str)
{
    for (int i = 0; i < NUM_KEYWORDS; i++)
    {
        if (strcmp(str, keywords[i]) == 0)
        {
            return 1;
        }
    }
    return 0;
}

int getIdentifierToken(FILE *f, TokenBuffer *buf)
{
    int c = getc(f);
    if (c == EOF)
        return EXIT_EOF;
    Token *identifier = initToken();
    while (isalpha(c))
    {
        identifier->tokenType = IDENTIFIER;
        identifier->tokenContent[identifier->size++] = c;
        c = getc(f);
    }
    if (isKeyword(identifier->tokenContent))
        identifier->tokenType = KEYWORD;
    if (identifier->tokenType == IDENTIFIER || identifier->tokenType == KEYWORD)
    {
        buf->tokenBuf[buf->count++] = identifier;
        return EXIT_TOKEN_CREATED;
    }
    else
    {
        freeToken(identifier);
        ungetc(c , f);
        return EXIT_TOKEN_FAILED;
    }
}

int getLiteralToken(FILE * f , TokenBuffer * buf){
    int c = getc(f);
    if(c == EOF) return EXIT_EOF;
    Token * literal = initToken();
    literal->tokenType = LITERAL;
    if(c == '\''){
        do{
            literal->tokenContent[literal->size++] = c;
        }while(c != '\'');
        literal->tokenContent[literal->size++] = c;
        buf->tokenBuf[buf->count++] = literal;
    }
    else if(c == '\"'){
        do{
            literal->tokenContent[literal->size++] = c;
        }while(c != '\"');
        literal->tokenContent[literal->size++] = c;
        buf->tokenBuf[buf->count++] = literal;
    }
    else{
        freeToken(literal);
        return EXIT_TOKEN_FAILED;
    }
}

void printTokenBuffer(TokenBuffer *buf)
{
    for (int i = 0; i < buf->count; i++)
    {
        if (buf->tokenBuf[i] != NULL)
            printToken(buf->tokenBuf[i]);
    }
}

int getNextToken(FILE *f)
{
    int c = getc(f);
    char skipChars[100] = " \t\n";
    if (c == EOF)
        return EXIT_EOF;
    while (strchr(skipChars, c))
    {
        c = getc(f);
        if (c == EOF)
            return EXIT_EOF;
    }
    int flag = EXIT_TOKEN_FAILED;
    TokenBuffer *buf = initTokenBuffer();
    while (flag != EXIT_EOF)
    {
        flag = getIdentifierToken(f, buf);
        if (flag == EXIT_TOKEN_FAILED)
            flag = getLiteralToken(f , buf);
        // else if (flag == EXIT_TOKEN_FAILED)
        //     flag = getLiteralToken(f);
        // else if (flag == EXIT_TOKEN_FAILED)
        //     flag = getLiteralToken(f);
        // else if (flag == EXIT_TOKEN_FAILED)
        //     flag = getNumberToken(f);
        // else if (flag == EXIT_TOKEN_FAILED)
        //     flag = getSeperatorToken(f);
        // else if (flag == EXIT_TOKEN_FAILED)
        //     flag = getRelOpToken(f);
        // else if (flag == EXIT_TOKEN_FAILED)
        //     flag = getArithmeticToken(f);
        // else if (flag == EXIT_TOKEN_FAILED)
        //     flag = getUnknownToken(f);
        // else
        //     flag = EXIT_EOF;
    }
    // printf("%c\n" , c);
    printTokenBuffer(buf);
    return EXIT_SUCCESS;
}

void lexer(FILE *f)
{
    int c = getc(f);
    if (c == EOF)
        return;
    ungetc(c, f);
    int flag = EXIT_SUCCESS;
    while (flag == EXIT_SUCCESS)
        flag = getNextToken(f);
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        printf("Execute Format : ./Lexer <filenameToTokenize.c>");
        return EXIT_FAILURE;
    }
    FILE *fp = fopen(argv[1], "r");
    lexer(fp);
    return 0;
}
