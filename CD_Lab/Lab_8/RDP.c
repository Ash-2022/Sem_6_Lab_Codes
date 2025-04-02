#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_TOKEN_LEN 100
#define MAX_TOKENS 1000

// Token types from lexer.c
enum {
    IDENTIFIER = 0,
    KEYWORD,
    LITERAL,
    NUMBER,
    SEPERATOR,  // Restored typo from "SEPARATOR" to match lexer
    RELOP,
    ARITHMETIC,
    UNKNOWN
};

// Token structure (similar to lexer.c)
typedef struct {
    char tokenContent[MAX_TOKEN_LEN];
    int tokenType;
    int row;
    int col;
} Token;

// Token buffer
Token tokens[MAX_TOKENS];
int tokenCount = 0;
int currentToken = 0;

// Function prototypes for recursive descent parser
void program();
void declarations();
void data_type();
void identifier_list();
void statement_list();
void statement();
void assign_stat();
void decision_stat();
void expr();
void simple_expr();
void eprime();
void term();
void seprime();
void factor(); // Modified to support char literals
void tprime();
void relop();
void addop();
void mulop();

// Error reporting function
void syntaxError(const char* message) {
    fprintf(stderr, "Syntax Error at line %d, column %d: %s\n", 
            tokens[currentToken].row, tokens[currentToken].col, message);
    fprintf(stderr, "Current token: %s\n", tokens[currentToken].tokenContent);
    exit(EXIT_FAILURE);
}

// Check if current token matches expected content
int match(int expectedType, const char* expectedContent) {
    if (currentToken >= tokenCount) {
        syntaxError("Unexpected end of input");
    }
    
    if (tokens[currentToken].tokenType == expectedType && 
        (expectedContent == NULL || strcmp(tokens[currentToken].tokenContent, expectedContent) == 0)) {
        currentToken++;
        return 1;
    }
    return 0;
}

// Require a match or report error
void expect(int expectedType, const char* expectedContent) {
    if (!match(expectedType, expectedContent)) {
        char error[200];
        sprintf(error, "Expected %s '%s'", 
                expectedType == IDENTIFIER ? "identifier" :
                expectedType == KEYWORD ? "keyword" :
                expectedType == SEPERATOR ? "seperator" :
                expectedType == RELOP ? "relational operator" :
                expectedType == ARITHMETIC ? "arithmetic operator" :
                "token",
                expectedContent ? expectedContent : "");
        syntaxError(error);
    }
}

// Parse a program
void program() {
    printf("Parsing program...\n");
    expect(IDENTIFIER, "main");
    expect(SEPERATOR, "(");
    expect(SEPERATOR, ")");
    expect(SEPERATOR, "{");
    declarations();
    statement_list();
    expect(SEPERATOR, "}");
    printf("Program parsed successfully!\n");
}

// Parse declarations
void declarations() {
    printf("Parsing declarations...\n");
    while (tokens[currentToken].tokenType == KEYWORD && 
           (strcmp(tokens[currentToken].tokenContent, "int") == 0 || 
            strcmp(tokens[currentToken].tokenContent, "char") == 0)) {
        data_type();
        identifier_list();
        expect(SEPERATOR, ";");  // Semicolon comes after the entire identifier list
    }
}

// Parse data type
void data_type() {
    printf("Parsing data type...\n");
    if (match(KEYWORD, "int") || match(KEYWORD, "char")) {
        // Successfully matched a data type
    } else {
        syntaxError("Expected data type (int or char)");
    }
}

// Parse identifier list
void identifier_list() {
    printf("Parsing identifier list...\n");
    
    // Must have at least one identifier
    if (tokens[currentToken].tokenType == IDENTIFIER) {
        currentToken++;
    } else {
        syntaxError("Expected identifier");
    }
    
    // Process multiple identifiers separated by commas
    while (currentToken < tokenCount && 
           tokens[currentToken].tokenType == SEPERATOR && 
           strcmp(tokens[currentToken].tokenContent, ",") == 0) {
        
        currentToken++; // Consume the comma
        
        if (tokens[currentToken].tokenType == IDENTIFIER) {
            currentToken++;
        } else {
            syntaxError("Expected identifier after comma");
        }
    }
}

// Parse statement list
void statement_list() {
    printf("Parsing statement list...\n");
    while (currentToken < tokenCount && 
           !(tokens[currentToken].tokenType == SEPERATOR && 
             strcmp(tokens[currentToken].tokenContent, "}") == 0) &&
           !(tokens[currentToken].tokenType == KEYWORD && 
             strcmp(tokens[currentToken].tokenContent, "else") == 0)) {
        statement();
    }
}

// Parse statement
void statement() {
    printf("Parsing statement...\n");
    if (tokens[currentToken].tokenType == IDENTIFIER) {
        assign_stat();
    } else if (tokens[currentToken].tokenType == KEYWORD && 
               strcmp(tokens[currentToken].tokenContent, "if") == 0) {
        decision_stat();
    } else {
        syntaxError("Expected statement (assignment or if statement)");
    }
}

// Parse assignment statement
void assign_stat() {
    printf("Parsing assignment statement...\n");
    expect(IDENTIFIER, NULL);
    expect(SEPERATOR, "=");
    expr();
    expect(SEPERATOR, ";");
}

// Parse decision statement (if-else)
void decision_stat() {
    printf("Parsing decision statement...\n");
    expect(KEYWORD, "if");
    expect(SEPERATOR, "(");
    expr();
    expect(SEPERATOR, ")");
    expect(SEPERATOR, "{");
    statement_list();
    expect(SEPERATOR, "}");
    
    // Optional else part
    if (match(KEYWORD, "else")) {
        expect(SEPERATOR, "{");
        statement_list();
        expect(SEPERATOR, "}");
    }
}

// Parse expression
void expr() {
    printf("Parsing expression...\n");
    simple_expr();
    eprime();
}

// Parse e'
void eprime() {
    printf("Parsing e'...\n");
    if (tokens[currentToken].tokenType == RELOP) {
        relop();
        simple_expr();
    }
    // else epsilon
}

// Parse simple expression
void simple_expr() {
    printf("Parsing simple expression...\n");
    term();
    seprime();
}

// Parse se'
void seprime() {
    printf("Parsing se'...\n");
    if (tokens[currentToken].tokenType == ARITHMETIC && 
        (strcmp(tokens[currentToken].tokenContent, "+") == 0 || 
         strcmp(tokens[currentToken].tokenContent, "-") == 0)) {
        addop();
        term();
        seprime();
    }
    // else epsilon
}

// Parse term
void term() {
    printf("Parsing term...\n");
    factor();
    tprime();
}

// Parse t'
void tprime() {
    printf("Parsing t'...\n");
    if (tokens[currentToken].tokenType == ARITHMETIC && 
        (strcmp(tokens[currentToken].tokenContent, "*") == 0 || 
         strcmp(tokens[currentToken].tokenContent, "/") == 0 || 
         strcmp(tokens[currentToken].tokenContent, "%") == 0)) {
        mulop();
        factor();
        tprime();
    }
    // else epsilon
}

// Parse factor
void factor() {
    printf("Parsing factor...\n");
    if (match(IDENTIFIER, NULL)) {
        // Successfully matched an identifier
    } else if (match(NUMBER, NULL)) {
        // Successfully matched a number
    } else if (match(LITERAL, NULL)) { // Added support for character literals
        // Successfully matched a character literal
    } else {
        syntaxError("Expected factor (identifier, number, or character literal)");
    }
}

// Parse relational operator
void relop() {
    printf("Parsing relational operator...\n");
    if (tokens[currentToken].tokenType == RELOP || 
        (tokens[currentToken].tokenType == SEPERATOR && 
         strcmp(tokens[currentToken].tokenContent, "=") == 0) ||
        (tokens[currentToken].tokenType == ARITHMETIC && 
         strcmp(tokens[currentToken].tokenContent, "!=") == 0)) {
        currentToken++;
    } else {
        syntaxError("Expected relational operator (==, !=, <, >, <=, >=)");
    }
}

// Parse additive operator
void addop() {
    printf("Parsing additive operator...\n");
    if (tokens[currentToken].tokenType == ARITHMETIC && 
        (strcmp(tokens[currentToken].tokenContent, "+") == 0 || 
         strcmp(tokens[currentToken].tokenContent, "-") == 0)) {
        currentToken++;
    } else {
        syntaxError("Expected additive operator (+ or -)");
    }
}

// Parse multiplicative operator
void mulop() {
    printf("Parsing multiplicative operator...\n");
    if (tokens[currentToken].tokenType == ARITHMETIC && 
        (strcmp(tokens[currentToken].tokenContent, "*") == 0 || 
         strcmp(tokens[currentToken].tokenContent, "/") == 0 || 
         strcmp(tokens[currentToken].tokenContent, "%") == 0)) {
        currentToken++;
    } else {
        syntaxError("Expected multiplicative operator (*, /, or %)");
    }
}

// Load tokens from a .tokens file
int loadTokens(const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error opening token file: %s\n", filename);
        return 0;
    }
    
    char line[256];
    tokenCount = 0;
    
    while (fgets(line, sizeof(line), fp) && tokenCount < MAX_TOKENS) {
        // Remove newline character
        line[strcspn(line, "\n")] = 0;
        
        // Parse token information from the line using a more robust approach
        if (line[0] == '<') {
            char *start = line + 1;  // Skip the opening '<'
            char *end;
            
            // Extract token type
            end = strchr(start, ',');
            if (!end) continue;
            
            char type[20];
            size_t len = end - start;
            if (len >= sizeof(type)) len = sizeof(type) - 1;
            strncpy(type, start, len);
            type[len] = '\0';
            
            // Skip the comma and space
            start = end + 2;
            
            // Extract token content
            end = strrchr(line, '>');
            if (!end) continue;
            
            // Work backwards to find the last two commas before '>'
            char *lastComma = end;
            int commasFound = 0;
            while (lastComma > start && commasFound < 2) {
                lastComma--;
                if (*lastComma == ',') commasFound++;
            }
            
            if (commasFound < 2) continue;
            
            // Extract the content between the first comma and the second-to-last comma
            char content[MAX_TOKEN_LEN];
            len = lastComma - start;
            if (len >= sizeof(content)) len = sizeof(content) - 1;
            strncpy(content, start, len);
            content[len] = '\0';
            
            // Extract row and column
            int row = 0, col = 0;
            sscanf(lastComma + 1, " %d, %d>", &row, &col);
            
            // Set token type based on the type string (rest of code same as before)
            if (strcmp(type, "identifier") == 0) {
                tokens[tokenCount].tokenType = IDENTIFIER;
            } else if (strcmp(type, "keyword") == 0) {
                tokens[tokenCount].tokenType = KEYWORD;
            } else if (strcmp(type, "literal") == 0) {
                tokens[tokenCount].tokenType = LITERAL;
            } else if (strcmp(type, "number") == 0) {
                tokens[tokenCount].tokenType = NUMBER;
            } else if (strcmp(type, "seperator") == 0) {
                tokens[tokenCount].tokenType = SEPERATOR;
            } else if (strcmp(type, "relop") == 0) {
                tokens[tokenCount].tokenType = RELOP;
            } else if (strcmp(type, "arithmetic") == 0) {
                tokens[tokenCount].tokenType = ARITHMETIC;
            } else {
                tokens[tokenCount].tokenType = UNKNOWN;
            }
            
            strncpy(tokens[tokenCount].tokenContent, content, MAX_TOKEN_LEN - 1);
            tokens[tokenCount].tokenContent[MAX_TOKEN_LEN - 1] = '\0';
            tokens[tokenCount].row = row;
            tokens[tokenCount].col = col;
            
            tokenCount++;
        }
    }
    
    fclose(fp);
    printf("Loaded %d tokens from %s\n", tokenCount, filename);
    return tokenCount > 0;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <tokens_file>\n", argv[0]);
        return EXIT_FAILURE;
    }
    
    if (!loadTokens(argv[1])) {
        fprintf(stderr, "Failed to load tokens from %s\n", argv[1]);
        return EXIT_FAILURE;
    }
    
    // Initialize parsing from the first token
    currentToken = 0;
    
    // Start recursive descent parsing
    program();
    
    // If we reach here without errors, the input program is syntactically correct
    printf("\nParsing completed successfully!\n");
    printf("The input program conforms to the specified grammar.\n");
    
    return EXIT_SUCCESS;
}
