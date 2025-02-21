#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

void preprocess(FILE* input, FILE* output)
{
    int c;
    int prev = '\0';
    int inString = 0;
    int stringChar = '\0';
    int suppressNewline = 0;  // Flag to suppress newlines after comments or directives
    
    while ((c = getc(input)) != EOF) {
        // Handle preprocessor directives
        if (c == '#' && !inString) {
            suppressNewline = 1;  // Suppress newline after directive
            while ((c = getc(input)) != EOF && c != '\n');
            if (c == EOF) {
                break;
            }
            continue;
        }

        // Handle strings
        if ((c == '"' || c == '\'') && prev != '\\' && !inString) {
            inString = 1;
            stringChar = c;
            putc(c, output);
        } else if (c == stringChar && prev != '\\' && inString) {
            inString = 0;
            putc(c, output);
        } else if (inString) {
            putc(c, output);
        }
        // Handle comments
        else if (c == '/' && !inString) {
            int next = getc(input);
            if (next == '/') {  // Single-line comment
                suppressNewline = 1;  // Suppress newline after comment
                while ((c = getc(input)) != EOF && c != '\n');
                if (c == EOF) {
                    break;
                }
            } else if (next == '*') {  // Multi-line comment
                while (1) {
                    c = getc(input);
                    if (c == EOF) {
                        break;
                    }
                    if (c == '*') {
                        next = getc(input);
                        if (next == '/') {
                            break;
                        }
                        if (next == EOF) {
                            break;
                        }
                        ungetc(next, input);
                    }
                }
            } else {
                putc('/', output);
                putc(next, output);
            }
        }
        // Handle newlines
        else if (c == '\n') {
            if (!suppressNewline) {
                putc(c, output);  // Write newline only if not suppressed
            }
            suppressNewline = 0;  // Reset suppression flag
        }
        // Handle whitespace
        else if (isspace(c)) {
            if (!isspace(prev)) {
                putc(' ', output);
            }
        }
        // Handle regular characters
        else {
            putc(c, output);
            suppressNewline = 0;  // Reset suppression flag for non-whitespace characters
        }
        
        prev = c;
    }
}

#endif
