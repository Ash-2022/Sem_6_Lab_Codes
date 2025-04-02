/*
Grammar : 
script → statement*
statement → for_loop | case_statement | other_command
for_loop → 'for' ID 'in' word* 'do' statement* 'done'
case_statement → 'case' word 'in' case_clause* 'esac'
case_clause → pattern ')' statement* ';;'
pattern → word ('|' word)*
word → (any non-whitespace, non-special characters)
*/
#include <stdio.h>
#include <ctype.h>
#include <string.h>

#define MAX_TOKENS 500
#define MAX_TOKEN_LEN 100

typedef enum {
    TOK_FOR, TOK_IN, TOK_DO, TOK_DONE,
    TOK_CASE, TOK_ESAC, TOK_LPAREN, TOK_RPAREN,
    TOK_PIPE, TOK_SEMICOLON, TOK_WORD, TOK_EOF
} TokenType;

typedef struct {
    TokenType type;
    char value[MAX_TOKEN_LEN];
    int line;
    int col;
} Token;

Token tokens[MAX_TOKENS];
int token_count = 0;
int current_token = 0;
int current_line = 1;
int current_col = 1;

void add_token(TokenType type, const char* value, int line, int col) {
    if (token_count < MAX_TOKENS) {
        tokens[token_count].type = type;
        strncpy(tokens[token_count].value, value, MAX_TOKEN_LEN-1);
        tokens[token_count].line = line;
        tokens[token_count].col = col;
        token_count++;
    }
}

void lexer(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) { perror("Error"); return; }

    char ch, word[MAX_TOKEN_LEN] = {0};
    int word_pos = 0;
    int word_line = 1, word_col = 1;
    
    while ((ch = fgetc(file)) != EOF) {
        current_col++;
        if (ch == '\n') { current_line++; current_col = 0; }

        if (isspace(ch)) {
            if (word_pos > 0) {
                word[word_pos] = '\0';
                // Check for keywords before adding as word
                if (strcmp(word, "for") == 0) add_token(TOK_FOR, word, word_line, word_col);
                else if (strcmp(word, "in") == 0) add_token(TOK_IN, word, word_line, word_col);
                else if (strcmp(word, "do") == 0) add_token(TOK_DO, word, word_line, word_col);
                else if (strcmp(word, "done") == 0) add_token(TOK_DONE, word, word_line, word_col);
                else if (strcmp(word, "case") == 0) add_token(TOK_CASE, word, word_line, word_col);
                else if (strcmp(word, "esac") == 0) add_token(TOK_ESAC, word, word_line, word_col);
                else add_token(TOK_WORD, word, word_line, word_col);
                word_pos = 0;
            }
            continue;
        }

        if (word_pos == 0) { word_line = current_line; word_col = current_col; }

        // Handle special characters
        if (ch == '(') { add_token(TOK_LPAREN, "(", current_line, current_col); continue; }
        if (ch == ')') { add_token(TOK_RPAREN, ")", current_line, current_col); continue; }
        if (ch == '|') { add_token(TOK_PIPE, "|", current_line, current_col); continue; }
        if (ch == ';') {
            if ((ch = fgetc(file)) == ';') {
                add_token(TOK_SEMICOLON, ";;", current_line, current_col);
                current_col++;
            } else {
                ungetc(ch, file);
                word[word_pos++] = ';';
            }
            continue;
        }

        word[word_pos++] = ch;
    }

    // Handle any remaining word at EOF
    if (word_pos > 0) {
        word[word_pos] = '\0';
        if (strcmp(word, "done") == 0) add_token(TOK_DONE, word, word_line, word_col);
        else if (strcmp(word, "esac") == 0) add_token(TOK_ESAC, word, word_line, word_col);
        else add_token(TOK_WORD, word, word_line, word_col);
    }
    add_token(TOK_EOF, "", current_line, current_col);
    fclose(file);
}

Token peek() { return tokens[current_token]; }
Token consume() { return tokens[current_token++]; }

void parse_word() {
    if (peek().type == TOK_WORD) {
        printf("%d:%d: WORD: %s\n", peek().line, peek().col, peek().value);
        consume();
    } else {
        printf("%d:%d: Error: Expected word, got %s\n", 
               peek().line, peek().col, peek().value);
        consume(); // Skip bad token to prevent infinite loop
    }
}

void parse_for_loop();
void parse_case_statement();

void parse_script() {
    while (peek().type != TOK_EOF) {
        if (peek().type == TOK_FOR) parse_for_loop();
        else if (peek().type == TOK_CASE) parse_case_statement();
        else if (peek().type == TOK_WORD) parse_word();
        else {
            printf("%d:%d: Unexpected token: %s\n", peek().line, peek().col, peek().value);
            consume();
        }
    }
}

void parse_for_loop() {
    Token t = peek();
    printf("%d:%d: FOR_LOOP_START\n", t.line, t.col);
    consume(); // 'for'
    parse_word(); // var
    if (peek().type != TOK_IN) { 
        printf("%d:%d: Error: Expected 'in'\n", peek().line, peek().col); 
        return; 
    }
    consume(); // 'in'
    while (peek().type == TOK_WORD) parse_word(); // list
    if (peek().type != TOK_DO) { 
        printf("%d:%d: Error: Expected 'do'\n", peek().line, peek().col); 
        return; 
    }
    consume(); // 'do'
    while (peek().type != TOK_DONE && peek().type != TOK_EOF) {
        if (peek().type == TOK_FOR) parse_for_loop();
        else if (peek().type == TOK_CASE) parse_case_statement();
        else if (peek().type == TOK_WORD) parse_word();
        else break;
    }
    if (peek().type != TOK_DONE) { 
        printf("%d:%d: Error: Expected 'done'\n", peek().line, peek().col); 
        return; 
    }
    printf("%d:%d: FOR_LOOP_END\n", peek().line, peek().col);
    consume(); // 'done'
}

void parse_case_statement() {
    Token t = peek();
    printf("%d:%d: CASE_START\n", t.line, t.col);
    consume(); // 'case'
    parse_word(); // expr
    if (peek().type != TOK_IN) { 
        printf("%d:%d: Error: Expected 'in'\n", peek().line, peek().col); 
        return; 
    }
    consume(); // 'in'
    while (peek().type != TOK_ESAC && peek().type != TOK_EOF) {
        while (peek().type == TOK_WORD || peek().type == TOK_PIPE) {
            if (peek().type == TOK_PIPE) consume();
            else parse_word();
        }
        if (peek().type != TOK_RPAREN) { 
            printf("%d:%d: Error: Expected ')'\n", peek().line, peek().col); 
            return; 
        }
        consume(); // ')'
        while (peek().type != TOK_SEMICOLON && peek().type != TOK_EOF) {
            if (peek().type == TOK_FOR) parse_for_loop();
            else if (peek().type == TOK_CASE) parse_case_statement();
            else if (peek().type == TOK_WORD) parse_word();
            else break;
        }
        if (peek().type != TOK_SEMICOLON) { 
            printf("%d:%d: Error: Expected ';;'\n", peek().line, peek().col); 
            return; 
        }
        consume(); // ';;'
    }
    if (peek().type != TOK_ESAC) { 
        printf("%d:%d: Error: Expected 'esac'\n", peek().line, peek().col); 
        return; 
    }
    printf("%d:%d: CASE_END\n", peek().line, peek().col);
    consume(); // 'esac'
}

int main(int argc, char* argv[]) {
    if (argc < 2) { printf("Usage: %s <script.sh>\n", argv[0]); return 1; }
    lexer(argv[1]);
    parse_script();
    return 0;
}
