#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_TOKEN_LEN 100
#define MAX_BUFFER_LEN 1000
#define SYMBOL_TABLE_SIZE 997 // Prime number for better hash distribution

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

// Symbol table entry structure
typedef struct SymbolEntry {
    char *name;                 // Identifier name
    int tokenType;              // Token type
    int line;                   // Line where first declared
    struct SymbolEntry *next;   // For hash collision handling
} SymbolEntry;

// Symbol table structure
typedef struct {
    SymbolEntry **entries;
    int size;
    int count;
} SymbolTable;

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
    int row;    // Line number
    int col;    // Column number
} Token;

typedef struct
{
    Token **tokenBuf;
    int count;
} TokenBuffer;

// Current line and column counters
int currentLine = 1;
int currentCol = 0;

// Output file for tokens
FILE *tokenOutputFile = NULL;

// Hash function for symbol table
unsigned int hash(const char *str) {
    unsigned int hash = 5381;
    int c;
    
    while ((c = *str++))
        hash = ((hash << 5) + hash) + c; // hash * 33 + c
    
    return hash;
}

// Initialize the symbol table
SymbolTable *initSymbolTable() {
    SymbolTable *table = (SymbolTable *)malloc(sizeof(SymbolTable));
    if (!table) {
        fprintf(stderr, "Memory allocation failed for symbol table\n");
        exit(EXIT_FAILURE);
    }
    
    table->size = SYMBOL_TABLE_SIZE;
    table->count = 0;
    table->entries = (SymbolEntry **)calloc(table->size, sizeof(SymbolEntry *));
    if (!table->entries) {
        fprintf(stderr, "Memory allocation failed for symbol table entries\n");
        free(table);
        exit(EXIT_FAILURE);
    }
    
    return table;
}

// Free symbol table memory
void freeSymbolTable(SymbolTable *table) {
    if (!table) return;
    
    for (int i = 0; i < table->size; i++) {
        SymbolEntry *entry = table->entries[i];
        while (entry) {
            SymbolEntry *temp = entry;
            entry = entry->next;
            free(temp->name);
            free(temp);
        }
    }
    
    free(table->entries);
    free(table);
}

// Insert or update a symbol in the symbol table
SymbolEntry *insertSymbol(SymbolTable *table, const char *name, int tokenType) {
    if (!table || !name) return NULL;
    
    unsigned int index = hash(name) % table->size;
    SymbolEntry *current = table->entries[index];
    
    // Check if the symbol already exists
    while (current) {
        if (strcmp(current->name, name) == 0) {
            return current; // Symbol already exists
        }
        current = current->next;
    }
    
    // Create a new symbol entry
    SymbolEntry *newEntry = (SymbolEntry *)malloc(sizeof(SymbolEntry));
    if (!newEntry) {
        fprintf(stderr, "Memory allocation failed for new symbol entry\n");
        return NULL;
    }
    
    newEntry->name = strdup(name);
    if (!newEntry->name) {
        fprintf(stderr, "Memory allocation failed for symbol name\n");
        free(newEntry);
        return NULL;
    }
    
    newEntry->tokenType = tokenType;
    newEntry->line = currentLine;
    newEntry->next = table->entries[index];
    table->entries[index] = newEntry;
    table->count++;
    
    return newEntry;
}

// Lookup a symbol in the symbol table
SymbolEntry *lookupSymbol(SymbolTable *table, const char *name) {
    if (!table || !name) return NULL;
    
    unsigned int index = hash(name) % table->size;
    SymbolEntry *current = table->entries[index];
    
    while (current) {
        if (strcmp(current->name, name) == 0) {
            return current;
        }
        current = current->next;
    }
    
    return NULL; // Symbol not found
}

// Print the symbol table
void printSymbolTable(SymbolTable *table) {
    if (!table) return;
    
    fprintf(tokenOutputFile, "\n--- SYMBOL TABLE ---\n");
    fprintf(tokenOutputFile, "%-20s %-15s %-10s\n", "IDENTIFIER", "TYPE", "LINE");
    fprintf(tokenOutputFile, "-----------------------------------------------------\n");
    
    for (int i = 0; i < table->size; i++) {
        SymbolEntry *entry = table->entries[i];
        while (entry) {
            fprintf(tokenOutputFile, "%-20s %-15s %-10d\n", 
                   entry->name, 
                   tokenType[entry->tokenType],
                   entry->line);
            entry = entry->next;
        }
    }
    fprintf(tokenOutputFile, "-----------------------------------------------------\n");
    fprintf(tokenOutputFile, "Total symbols: %d\n", table->count);
}

Token *initToken()
{
    Token *token = (Token *)malloc(sizeof(Token));
    if (!token) {
        fprintf(stderr, "Memory allocation failed for token\n");
        exit(EXIT_FAILURE);
    }
    
    token->tokenContent = (char *)malloc(sizeof(char) * MAX_TOKEN_LEN);
    if (!token->tokenContent) {
        fprintf(stderr, "Memory allocation failed for token content\n");
        free(token);
        exit(EXIT_FAILURE);
    }
    
    token->size = 0;
    token->tokenType = UNKNOWN;
    token->row = currentLine;
    token->col = currentCol;
    memset(token->tokenContent, '\0', MAX_TOKEN_LEN);
    return token;
}

void freeToken(Token *tok)
{
    if (!tok) return;
    free(tok->tokenContent);
    free(tok);
}

void freeBuffer(TokenBuffer *buf)
{
    if (!buf) return;
    
    for (int i = 0; i < buf->count; i++)
        freeToken(buf->tokenBuf[i]);
    free(buf->tokenBuf);
    free(buf);
}

void printToken(Token *tok)
{
    if (!tok) return;
    fprintf(tokenOutputFile, "<%s, %s, %d, %d>\n", 
           tokenType[tok->tokenType], 
           tok->tokenContent, 
           tok->row, 
           tok->col);
}

TokenBuffer *initTokenBuffer()
{
    TokenBuffer *tokenBuf = (TokenBuffer *)malloc(sizeof(TokenBuffer));
    if (!tokenBuf) {
        fprintf(stderr, "Memory allocation failed for token buffer\n");
        exit(EXIT_FAILURE);
    }
    
    tokenBuf->tokenBuf = (Token **)malloc(sizeof(Token *) * MAX_BUFFER_LEN);
    if (!tokenBuf->tokenBuf) {
        fprintf(stderr, "Memory allocation failed for token buffer array\n");
        free(tokenBuf);
        exit(EXIT_FAILURE);
    }
    
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

int getIdentifierToken(FILE *f, TokenBuffer *buf, SymbolTable *symTable)
{
    int c = getc(f);
    if (c == EOF)
        return EXIT_EOF;
    if (!isalpha(c) && c != '_')
    {
        ungetc(c, f);
        return EXIT_TOKEN_FAILED;
    }
    Token *identifier = initToken();
    int startCol = currentCol;
    do
    {
        identifier->tokenType = IDENTIFIER;
        identifier->tokenContent[identifier->size++] = c;
        currentCol++;
        c = getc(f);
    } while (isalnum(c) || c == '_');
    
    identifier->col = startCol;  // Set to the starting column
    
    if (isKeyword(identifier->tokenContent))
        identifier->tokenType = KEYWORD;
    
    if (identifier->tokenType == IDENTIFIER || identifier->tokenType == KEYWORD)
    {
        // Add identifier to symbol table if it's not a keyword
        if (identifier->tokenType == IDENTIFIER) {
            insertSymbol(symTable, identifier->tokenContent, IDENTIFIER);
        }
        
        buf->tokenBuf[buf->count++] = identifier;
        ungetc(c, f);
        return EXIT_TOKEN_CREATED;
    }
    else
    {
        freeToken(identifier);
        ungetc(c, f);
        return EXIT_TOKEN_FAILED;
    }
}

int getLiteralToken(FILE *f, TokenBuffer *buf)
{
    int c = getc(f);
    if (c == EOF)
        return EXIT_EOF;
    Token *literal = initToken();
    int startCol = currentCol;
    literal->tokenType = LITERAL;
    if (c == '\'')
    {
        do
        {
            literal->tokenContent[literal->size++] = c;
            currentCol++;
            c = getc(f);
            // Handle escaped quotes
            if (c == '\\') {
                literal->tokenContent[literal->size++] = c;
                currentCol++;
                c = getc(f);
                literal->tokenContent[literal->size++] = c;
                currentCol++;
                c = getc(f);
            }
        } while (c != '\'' && c != EOF && c != '\n');
        
        if (c == EOF || c == '\n') {
            freeToken(literal);
            if (c == '\n') {
                currentLine++;
                currentCol = 0;
                ungetc(c, f);
            }
            return c == EOF ? EXIT_EOF : EXIT_TOKEN_FAILED;
        }
        
        literal->tokenContent[literal->size++] = c;
        currentCol++;
        literal->col = startCol;  // Set to the starting column
        buf->tokenBuf[buf->count++] = literal;
        return EXIT_TOKEN_CREATED;
    }
    else if (c == '\"')
    {
        do
        {
            literal->tokenContent[literal->size++] = c;
            currentCol++;
            c = getc(f);
            // Handle escaped quotes
            if (c == '\\') {
                literal->tokenContent[literal->size++] = c;
                currentCol++;
                c = getc(f);
                literal->tokenContent[literal->size++] = c;
                currentCol++;
                c = getc(f);
            }
        } while (c != '\"' && c != EOF && c != '\n');
        
        if (c == EOF || c == '\n') {
            freeToken(literal);
            if (c == '\n') {
                currentLine++;
                currentCol = 0;
                ungetc(c, f);
            }
            return c == EOF ? EXIT_EOF : EXIT_TOKEN_FAILED;
        }
        
        literal->tokenContent[literal->size++] = c;
        currentCol++;
        literal->col = startCol;  // Set to the starting column
        buf->tokenBuf[buf->count++] = literal;
        return EXIT_TOKEN_CREATED;
    }
    else
    {
        freeToken(literal);
        ungetc(c, f);
        return EXIT_TOKEN_FAILED;
    }
}

int getSeperatorToken(FILE *f, TokenBuffer *buf)
{
    int c = getc(f);
    char seperators[] = ",(.){;}";
    Token *seperator = initToken();
    int startCol = currentCol;
    seperator->tokenType = SEPERATOR;
    if (strchr(seperators, c))
    {
        seperator->tokenContent[seperator->size++] = c;
        currentCol++;
        seperator->col = startCol;  // Set to the starting column
        buf->tokenBuf[buf->count++] = seperator;
        return EXIT_TOKEN_CREATED;
    }
    else
    {
        freeToken(seperator);
        ungetc(c, f);
        return EXIT_TOKEN_FAILED;
    }
}

int getRelOpToken(FILE *f, TokenBuffer *buf)
{
    int c = getc(f);
    char seperators[] = "><";
    Token *relop = initToken();
    int startCol = currentCol;
    relop->tokenType = RELOP;
    if (strchr(seperators, c))
    {
        relop->tokenContent[relop->size++] = c;
        currentCol++;
        c = getc(f);
        if (c == '=')
        {
            relop->tokenContent[relop->size++] = c;
            currentCol++;
        }
        else
        {
            ungetc(c, f);
        }
        relop->col = startCol;  // Set to the starting column
        buf->tokenBuf[buf->count++] = relop;
        return EXIT_TOKEN_CREATED;
    }
    else if (c == '=')
    {
        relop->tokenContent[relop->size++] = c;
        currentCol++;
        c = getc(f);
        if (c == '=')
        {
            relop->tokenContent[relop->size++] = c;
            currentCol++;
        }
        else
        {
            ungetc(c, f);
            relop->tokenType = SEPERATOR;
        }
        relop->col = startCol;  // Set to the starting column
        buf->tokenBuf[buf->count++] = relop;
        return EXIT_TOKEN_CREATED;
    }
    else
    {
        freeToken(relop);
        ungetc(c, f);
        return EXIT_TOKEN_FAILED;
    }
}

int getArithmeticToken(FILE *f, TokenBuffer *buf)
{
    int c = getc(f);
    char operators[] = "+-*";
    Token *arithmetic = initToken();
    int startCol = currentCol;
    arithmetic->tokenType = ARITHMETIC;
    if (strchr(operators, c) || c == '%')
    {
        arithmetic->tokenContent[arithmetic->size++] = c;
        currentCol++;
        c = getc(f);
        if (strchr(operators, c) || c == '=')
        {
            arithmetic->tokenContent[arithmetic->size++] = c;
            currentCol++;
        }
        else
            ungetc(c, f);
        arithmetic->col = startCol;  // Set to the starting column
        buf->tokenBuf[buf->count++] = arithmetic;
        return EXIT_TOKEN_CREATED;
    }
    else if (c == '/')
    {
        arithmetic->tokenContent[arithmetic->size++] = c;
        currentCol++;
        c = getc(f);
        if (c == '=')
        {
            arithmetic->tokenContent[arithmetic->size++] = c;
            currentCol++;
        }
        else
            ungetc(c, f);
        arithmetic->col = startCol;  // Set to the starting column
        buf->tokenBuf[buf->count++] = arithmetic;
        return EXIT_TOKEN_CREATED;
    }
    else if (c == '!' || c == '&'){
        arithmetic->tokenContent[arithmetic->size++] = c;
        currentCol++;
        c = getc(f);
        if (c == '=')
        {
            arithmetic->tokenContent[arithmetic->size++] = c;
            currentCol++;
        }
        else
            ungetc(c, f);
        arithmetic->col = startCol;  // Set to the starting column
        buf->tokenBuf[buf->count++] = arithmetic;
        return EXIT_TOKEN_CREATED;
    }
    else
    {
        freeToken(arithmetic);
        ungetc(c, f);
        return EXIT_TOKEN_FAILED;
    }
}

int getNumberToken(FILE *f, TokenBuffer *buf)
{
    int c = getc(f);
    if (c == EOF)
        return EXIT_EOF;
    Token *number = initToken();
    int startCol = currentCol;
    if (isdigit(c) || c == '-')
    {
        do
        {
            number->tokenType = NUMBER;
            number->tokenContent[number->size++] = c;
            currentCol++;
            c = getc(f);
        } while (isdigit(c) || c == '.');
        number->col = startCol;  // Set to the starting column
        buf->tokenBuf[buf->count++] = number;
        ungetc(c, f);
        return EXIT_TOKEN_CREATED;
    }
    else
    {
        freeToken(number);
        ungetc(c, f);
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

int getNextToken(FILE *f, TokenBuffer *buf, SymbolTable *symTable)
{
    int c;
    char skipChars[] = " \t";

    while (1)
    {
        // Skip whitespace
        do
        {
            c = getc(f);
            if (c == EOF)
                return EXIT_EOF;
            if (c == '\n') {
                currentLine++;
                currentCol = 0;
            } else if (c == '\t') {
                currentCol += 4; // Assuming tab = 4 spaces
            } else {
                currentCol++;
            }
        } while (strchr(skipChars, c) || c == '\n');

        // Handle comments and directives
        if (c == '/')
        {
            int next = getc(f);
            currentCol++;

            // Single line comment
            if (next == '/')
            {
                currentCol++;
                while ((c = getc(f)) != EOF && c != '\n')
                    currentCol++;
                if (c == '\n') {
                    currentLine++;
                    currentCol = 0;
                }
                if (c == EOF)
                    return EXIT_EOF;
                continue; // Continue the outer loop to get the next real token
            }
            // Multi-line comment
            else if (next == '*')
            {
                currentCol++;
                int prev = 0;
                while ((c = getc(f)) != EOF)
                {
                    if (c == '\n') {
                        currentLine++;
                        currentCol = 0;
                    } else {
                        currentCol++;
                    }
                    if (prev == '*' && c == '/')
                        break;
                    prev = c;
                }
                if (c == EOF)
                    return EXIT_EOF;
                continue; // Continue the outer loop to get the next real token
            }
            else
            {
                // Not a comment, put back both characters for token processing
                ungetc(next, f);
                ungetc(c, f);
                currentCol -= 2;  // Adjust column counter
                break; // Break the loop to process this as a token
            }
        }
        // Handle preprocessor directives
        else if (c == '#')
        {
            while ((c = getc(f)) != EOF && c != '\n')
                currentCol++;
            if (c == '\n') {
                currentLine++;
                currentCol = 0;
            }
            if (c == EOF)
                return EXIT_EOF;
            continue; // Continue the outer loop to get the next real token
        }
        else
        {
            // Not a special case - put back the character and proceed to token matching
            ungetc(c, f);
            currentCol--;  // Adjust column counter
            break;
        }
    }

    // Save the starting column position
    int startCol = currentCol;

    // Try to match each token type
    int flag = getIdentifierToken(f, buf, symTable);
    if (flag == EXIT_TOKEN_FAILED)
        flag = getLiteralToken(f, buf);
    if (flag == EXIT_TOKEN_FAILED)
        flag = getNumberToken(f, buf);
    if (flag == EXIT_TOKEN_FAILED)
        flag = getSeperatorToken(f, buf);
    if (flag == EXIT_TOKEN_FAILED)
        flag = getRelOpToken(f, buf);
    if (flag == EXIT_TOKEN_FAILED)
        flag = getArithmeticToken(f, buf);

    if (flag == EXIT_TOKEN_FAILED)
    {
        // Handle unrecognized character - consume it and print a message
        c = getc(f);
        currentCol++;
        fprintf(stderr, "Skipping unrecognized character: %c (ASCII: %d) at line %d, column %d\n", 
                c, c, currentLine, currentCol-1);
        return EXIT_TOKEN_CREATED; // Continue processing
    }

    return flag;
}

void lexer(FILE *f, FILE *outfile)
{
    TokenBuffer *buf = initTokenBuffer();
    SymbolTable *symTable = initSymbolTable();
    int flag = EXIT_SUCCESS;

    // Reset line and column counters
    currentLine = 1;
    currentCol = 0;
    
    // Set the global output file
    tokenOutputFile = outfile;

    while (flag != EXIT_EOF)
    {
        flag = getNextToken(f, buf, symTable);

        if (buf->count >= MAX_BUFFER_LEN || flag == EXIT_EOF)
        {
            printTokenBuffer(buf);
            buf->count = 0;
        }
    }

    // Print the symbol table
    printSymbolTable(symTable);

    // Free resources
    freeBuffer(buf);
    freeSymbolTable(symTable);
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        printf("Execute Format : ./Lexer <filenameToTokenize.c>\n");
        return EXIT_FAILURE;
    }
    
    FILE *fp = fopen(argv[1], "r");
    if (!fp) {
        fprintf(stderr, "Error opening input file: %s\n", argv[1]);
        return EXIT_FAILURE;
    }
    
    // Create output filename by adding ".tokens" extension
    char outFileName[256];
    snprintf(outFileName, sizeof(outFileName), "%s.tokens", argv[1]);
    
    FILE *outfile = fopen(outFileName, "w");
    if (!outfile) {
        fprintf(stderr, "Error creating output file: %s\n", outFileName);
        fclose(fp);
        return EXIT_FAILURE;
    }
    
    printf("Analyzing file: %s\n", argv[1]);
    printf("Tokens will be written to: %s\n", outFileName);
    
    lexer(fp, outfile);
    
    // Close file handles
    fclose(fp);
    fclose(outfile);
    
    printf("Token analysis complete!\n");
    
    return 0;
}
