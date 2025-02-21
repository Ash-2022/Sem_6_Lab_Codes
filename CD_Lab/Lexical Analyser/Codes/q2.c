// #include <stdio.h>
// #include <string.h>
// #include <ctype.h>
// #include <stdlib.h>
// #define MAX_BUF_SIZE 100
// typedef enum {
//     KEYWORD,
//     IDENTIFIER,
//     OPERATOR,
//     NUMERIC_CONSTANT,
//     STRING_LITERAL,
//     SPECIAL_SYMBOL,
//     COMMENT,
//     PREPROCESSOR,
//     UNKNOWN,
//     EOF_TOKEN
// } TokenType;
// typedef struct {
//     int row;              
//     int column;           
//     TokenType type;       
//     char value[MAX_BUF_SIZE]; 
// } Token;
// void identify_keywords(char *word, Token *token);
// void identify_identifiers(char *word, Token *token);
// void identify_operators(char c, Token *token);
// void identify_special_symbols(char c, Token *token);
// Token getNextToken(FILE *fp);
// int main() {
//     FILE *fp = fopen("test.c", "r");  
//     if (fp == NULL) {
//         printf("Cannot open file\n");
//         return 1;
//     }
//     Token token;
//     while ((token = getNextToken(fp)).type != EOF_TOKEN) {
//         printf("Line: %d, Column: %d, Type: %d, Value: %s\n", token.row, token.column, token.type, token.value);
//     }
//     fclose(fp); 
//     return 0;
// }
// Token getNextToken(FILE *fp) {
//     Token token;
//     token.type = UNKNOWN;
//     token.value[0] = '\0';
//     int row = 1, column = 1;
//     char c;
//     int i = 0;

//     while ((c = fgetc(fp)) != EOF) {
//         if (isspace(c)) {
//             if (c == '\n') {
//                 row++;
//                 column = 1;
//             } else {
//                 column++;
//             }
//             continue;
//         }
//         if (c == '/' && (c = fgetc(fp)) == '/') {
//             while (c != '\n' && c != EOF) {
//                 c = fgetc(fp);
//             }
//             row++;
//             column = 1;
//             continue;
//         }
//         if (c == '/' && (c = fgetc(fp)) == '*') {
//             while (c != '*' || (c = fgetc(fp)) != '/') {
//                 c = fgetc(fp);
//             }
//             column++;
//             continue;
//         }
//         if (c == '#') {
//             while (c != '\n' && c != EOF) {
//                 c = fgetc(fp);
//             }
//             row++;
//             column = 1;
//             continue;
//         }
//         if (c == '"') {
//             token.type = STRING_LITERAL;
//             token.value[i++] = c;
//             c = fgetc(fp);
//             while (c != '"' && c != EOF) {
//                 token.value[i++] = c;
//                 c = fgetc(fp);
//             }
//             token.value[i++] = c; 
//             token.value[i] = '\0';
//             column += i;
//             return token;
//         }
//         if (c == '+' || c == '-' || c == '*' || c == '/' || c == '%' ||
//             c == '=' || c == '<' || c == '>' || c == '!' || c == '&' || c == '|') {
//             identify_operators(c, &token);
//             column++;
//             return token;
//         }

//         if (c == '(' || c == ')' || c == '{' || c == '}' || c == ',' || c == ';' ||
//             c == '[' || c == ']') {
//             identify_special_symbols(c, &token);
//             column++;
//             return token;
//         }
//         if (isalpha(c) || c == '_') {
//             token.value[i++] = c;
//             c = fgetc(fp);
//             while (isalnum(c) || c == '_') {
//                 token.value[i++] = c;
//                 c = fgetc(fp);
//             }
//             ungetc(c, fp);
//             token.value[i] = '\0';
//             identify_keywords(token.value, &token);
//             identify_identifiers(token.value, &token);
//             column += i;
//             return token;
//         }
//         if (isdigit(c)) {
//             token.value[i++] = c;
//             c = fgetc(fp);
//             while (isdigit(c)) {
//                 token.value[i++] = c;
//                 c = fgetc(fp);
//             }
//             ungetc(c, fp);
//             token.value[i] = '\0';
//             token.type = NUMERIC_CONSTANT;
//             column += i;
//             return token;
//         }
//         token.type = UNKNOWN;
//         token.value[0] = c;
//         token.value[1] = '\0';
//         column++;
//         return token;
//     }
//     token.type = EOF_TOKEN;
//     return token;
// }

// void identify_keywords(char *word, Token *token) {
//     const char *keywords[] = {
//         "int", "char", "void", "float", "if", "else", "while", "for", "return", "break", "continue", 
//         "switch", "case", "default", "typedef", "struct"
//     };
    
//     for (int i = 0; i < sizeof(keywords) / sizeof(keywords[0]); i++) {
//         if (strcmp(word, keywords[i]) == 0) {
//             token->type = KEYWORD;
//             return;
//         }
//     }
//     token->type = IDENTIFIER;
// }
// void identify_identifiers(char *word, Token *token) {
//     if (token->type == IDENTIFIER) {
//         strcpy(token->value, word);
//     }
// }
// void identify_operators(char c, Token *token) {
//     token->type = OPERATOR;
//     token->value[0] = c;
//     token->value[1] = '\0';
// }
// void identify_special_symbols(char c, Token *token) {
//     token->type = SPECIAL_SYMBOL;
//     token->value[0] = c;
//     token->value[1] = '\0';
// }
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>

#define MAX_BUF_SIZE 100

typedef enum {
    KEYWORD,
    IDENTIFIER,
    OPERATOR,
    NUMERIC_CONSTANT,
    STRING_LITERAL,
    SPECIAL_SYMBOL,
    COMMENT,
    PREPROCESSOR,
    UNKNOWN,
    EOF_TOKEN
} TokenType;

typedef struct {
    int row;              // Line number where token was found
    int column;           // Column number where token was found
    TokenType type;       // Type of token
    char value[MAX_BUF_SIZE]; // Token value
} Token;

void identify_keywords(char *word, Token *token);
void identify_identifiers(char *word, Token *token);
void identify_operators(char c, Token *token);
void identify_special_symbols(char c, Token *token);
Token getNextToken(FILE *fp);

int main() {
    FILE *fp = fopen("test.c", "r");  // Open a C source file for reading
    if (fp == NULL) {
        printf("Cannot open file\n");
        return 1;
    }

    Token token;
    while ((token = getNextToken(fp)).type != EOF_TOKEN) {
        printf("Line: %d, Column: %d, Type: %d, Value: %s\n", token.row, token.column, token.type, token.value);
    }

    fclose(fp); 
    return 0;
}

Token getNextToken(FILE *fp) {
    Token token;
    token.type = UNKNOWN;
    token.value[0] = '\0';
    int row = 1, column = 1;
    char c;
    int i = 0;

    while ((c = fgetc(fp)) != EOF) {
        if (isspace(c)) {
            if (c == '\n') {
                row++;
                column = 1;
            } else {
                column++;
            }
            continue;
        }

        // Handling single-line comments (//)
        if (c == '/' && (c = fgetc(fp)) == '/') {
            while (c != '\n' && c != EOF) {
                c = fgetc(fp);
            }
            row++;
            column = 1;
            continue;
        }

        // Handling multi-line comments (/* */)
        if (c == '/' && (c = fgetc(fp)) == '*') {
            while (c != '*' || (c = fgetc(fp)) != '/') {
                if (c == EOF) {
                    token.type = COMMENT;
                    return token;
                }
                c = fgetc(fp);
            }
            column++;
            continue;
        }

        // Handling preprocessor directives (#)
        if (c == '#') {
            while (c != '\n' && c != EOF) {
                c = fgetc(fp);
            }
            row++;
            column = 1;
            continue;
        }

        // Handling string literals (")
        if (c == '"') {
            token.type = STRING_LITERAL;
            token.value[i++] = c;
            c = fgetc(fp);
            while (c != '"' && c != EOF) {
                token.value[i++] = c;
                c = fgetc(fp);
            }
            token.value[i++] = c;  // Add closing quote
            token.value[i] = '\0';
            column += i;
            return token;
        }

        // Handling operators (+, -, *, /, %, etc.)
        if (c == '+' || c == '-' || c == '*' || c == '/' || c == '%' ||
            c == '=' || c == '<' || c == '>' || c == '!' || c == '&' || c == '|') {
            identify_operators(c, &token);
            column++;
            return token;
        }

        // Handling special symbols (e.g., parentheses, braces, semicolons, etc.)
        if (c == '(' || c == ')' || c == '{' || c == '}' || c == ',' || c == ';' ||
            c == '[' || c == ']') {
            identify_special_symbols(c, &token);
            column++;
            return token;
        }

        // Handling identifiers and keywords (e.g., int, float, myVar, etc.)
        if (isalpha(c) || c == '_') {
            token.value[i++] = c;
            c = fgetc(fp);
            while (isalnum(c) || c == '_') {
                token.value[i++] = c;
                c = fgetc(fp);
            }
            ungetc(c, fp);
            token.value[i] = '\0';
            identify_keywords(token.value, &token);
            identify_identifiers(token.value, &token);
            column += i;
            return token;
        }

        // Handling numeric constants (e.g., 123, 456, etc.)
        if (isdigit(c)) {
            token.value[i++] = c;
            c = fgetc(fp);
            while (isdigit(c)) {
                token.value[i++] = c;
                c = fgetc(fp);
            }
            ungetc(c, fp);
            token.value[i] = '\0';
            token.type = NUMERIC_CONSTANT;
            column += i;
            return token;
        }

        // If nothing else, treat it as an unknown character
        token.type = UNKNOWN;
        token.value[0] = c;
        token.value[1] = '\0';
        column++;
        return token;
    }

    token.type = EOF_TOKEN;
    return token;
}

void identify_keywords(char *word, Token *token) {
    const char *keywords[] = {
        "int", "char", "void", "float", "if", "else", "while", "for", "return", "break", "continue", 
        "switch", "case", "default", "typedef", "struct"
    };

    for (int i = 0; i < sizeof(keywords) / sizeof(keywords[0]); i++) {
        if (strcmp(word, keywords[i]) == 0) {
            token->type = KEYWORD;
            return;
        }
    }
    token->type = IDENTIFIER;
}

void identify_identifiers(char *word, Token *token) {
    if (token->type == IDENTIFIER) {
        strcpy(token->value, word);
    }
}

void identify_operators(char c, Token *token) {
    token->type = OPERATOR;
    token->value[0] = c;
    token->value[1] = '\0';
}

void identify_special_symbols(char c, Token *token) {
    token->type = SPECIAL_SYMBOL;
    token->value[0] = c;
    token->value[1] = '\0';
}
