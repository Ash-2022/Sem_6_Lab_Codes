#include<stdio.h>
#include<string.h>
#include<stdlib.h>
int main()
{
	char c,buf[10];
	FILE *fp=fopen("test.c","r");
	c = fgetc(fp);
	if (fp == NULL)
	{
		printf("Cannot open file \n");
		exit(0);
	}
	while(c!=EOF)
	{
		int i=0;
		buf[0]='\0';
		if(c=='=')
		{
			buf[i++]=c;
			c = fgetc(fp);
			if(c=='=')
			{
				buf[i++]=c;
				buf[i]='\0';
				printf("\n Relational operator : %s",buf);
			}
			else
			{
				buf[i]='\0';
			}
		}
		else if(c=='+' || c=='-' || c=='*' || c=='/' || c=='%')
		{
			buf[i++] = c;
			c = fgetc(fp);
			if(c == '+' || c=='-')
			{
				buf[i++]=c;
				buf[i]='\0';
				printf("\n Incremental operators : %s",buf);
			}
			else
			{
				buf[i] = '\0';
				printf("\n Aritmetic operator : %s",buf);
			}
		}
		else if(c == '&')
		{
			buf[i++] = c;
			c = fgetc(fp);
			if(c == '&')
			{
				buf[i++] = c;
				buf[i] = '\0';
				printf("\n Logical operator: %s",buf);
			}
			else
			{
				buf[i] = '\0';
				printf("\n Bitwise Operator: %s",buf);
			}
		}
		else if(c == '|')
		{
			buf[i++] = c;
			c = fgetc(fp);
			if(c == '|')
			{
				buf[i++] = c;
				buf[i] = '\0';
				printf("\n Logical operator: %s",buf);
			}
			else
			{
				buf[i] = '\0';
				printf("\n Bitwise Operator: %s",buf);
			}
		}
		else if(c == '!')
		{
			buf[i++] = c;
			c = fgetc(fp);
			if(c != '=')
			{
				buf[i] = '\0';
				printf("\n Logical operator : %s",buf);
			}
			else
			{
				buf[i++] = c;
				printf("\n Relational operator : %s",buf);
			}
		}
		else
		{
			if(c=='<'||c=='>')
			{
				buf[i++]=c;
				c = fgetc(fp);
				if(c=='=')
				{
					buf[i++]=c;
				}
				buf[i]='\0';
				printf("\n Relational operator : %s",buf);
			}
			else
			{
				buf[i]='\0';
			}
		}
		c = fgetc(fp);
	}}