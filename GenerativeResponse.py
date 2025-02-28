import string

from openai import OpenAI

def getKey(path: string):
    f = open(path, 'r')
    return f.readline()
def cleanResponse(statement: string):
