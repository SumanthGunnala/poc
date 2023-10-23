import javalang
import os
from sentence_transformers import SentenceTransformer
import re
import pandas as pd
import tokenize
from io import BytesIO
from collections import deque
import linecache
import re
import pandas as pd
import numpy as np
from nltk.cluster import KMeansClusterer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

model = SentenceTransformer('flax-sentence-embeddings/st-codesearch-distilroberta-base')

column_names=['Directory','Filename','Class Name','Class Name embeddings','Parent Class','Parent Class embeddings','Function Name','Function Name embeddings','Function Parameters','Function Parameter embeddings','Function Body','Function Body embeddings']
func_df_final = pd.DataFrame(columns=column_names)

def get_start_end_for_node(node_to_find, tree):
    start = None
    end = None
    for path, node in tree:
        if start is not None and node_to_find not in path:
            end = node.position
            return start, end
        if start is None and node == node_to_find:
            start = node.position
    return start, end

def get_string(start, end, data):
    if start is None:
        return ""

    # positions are all offset by 1. e.g. first line -> lines[0], start.line = 1
    end_pos = None

    if end is not None:
        end_pos = end.line - 1

    lines = data.splitlines(True)
    string = "".join(lines[start.line:end_pos])
    string = lines[start.line - 1] + string

    # When the method is the last one, it will contain a additional brace
    if end is None:
        left = string.count("{")
        right = string.count("}")
        if right - left == 1:
            p = string.rfind("}")
            string = string[:p]

    return string
def get_java_file_paths(directory):
    java_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".java"):
                file_path = os.path.join(root, file)
                java_files.append(file_path)

    return java_files

def get_file_and_directory_names(file_paths):
    file_and_directory_names = []

    for file_path in file_paths:
        directory_name = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        file_and_directory_names.append((file_path, file_name, directory_name))

    return file_and_directory_names

def read_java_file(file_path):
    with open(file_path, 'r') as file:
        java_code = file.read()
    return java_code


def extract_functions(java_code):
    tree = javalang.parse.parse(java_code)
    
    functions = []
    
    # Initialize class_name and parent_class with default values
    class_name = None
    parent_class = None
    
    for path, node in tree:
        if isinstance(node, javalang.tree.ClassDeclaration):
            class_name = node.name
            parent_class = node.extends
    
    for _, node in tree.filter(javalang.tree.MethodDeclaration):
        start, end = get_start_end_for_node(node, tree)

        function_name = node.name
        function_parameters = [param.type.name for param in node.parameters]
        function_body = get_string(start, end, java_code)

        # Check if '{' exists in the function body
        if '{' not in function_body:
            continue

        index = function_body.index('{')

        last_index = function_body.rfind('}')

        function_body = function_body[index: last_index + 1]
        
        functions.append((class_name, parent_class, function_name, function_parameters, function_body))

    return functions


#directory_name = "Project-5-CS-180-main"
directory_name = "innovation_coe"

directory_path = os.path.join(os.getcwd(), directory_name)

#directory_path = "D:/Nissan_copy/Project-5-CS-180-main"

java_file_paths = get_java_file_paths(directory_path)

file_and_directory_names = get_file_and_directory_names(java_file_paths)

# Print file paths, file names, and directory names
for file_path, file_name, directory_name in file_and_directory_names:
    
   
    java_code=read_java_file(file_path)

    
    result = extract_functions(java_code)
    dir_name_list=[]
    filename_list=[]
    class_name_list=[]
    parent_class_list=[]
    function_name_list=[]
    function_param_list=[]
    function_body_list=[]
    for class_name, parent_class, function_name, function_parameters, function_body in result:
        dir_name_list.append(directory_name)
        filename_list.append(file_name)
        if class_name==None:
            class_name_list.append('')
        else:
            class_name_list.append(class_name)
        if parent_class==None:
            parent_class_list.append('')
        else:
            parent_class_list.append(parent_class)
        if function_name==None:
            function_name_list.append('')
        else:
            function_name_list.append(function_name)
        if function_parameters==None:
            function_param_list.append('')
        else:
            params=''
            for param in function_parameters:
                params+=param+', '

            function_param_list.append(params)
        count=0
        function_body_list.append(function_body)


    class_name_embeddings = model.encode(class_name_list)
    class_name_embeddings_list=[]
    for embedding in class_name_embeddings:
        class_name_embeddings_list.append([embedding])
    # parent_class_embeddings = model.encode(parent_class_list)
    # parent_class_embeddings_list=[]
    # for embedding in parent_class_embeddings:
    #     parent_class_embeddings_list.append([embedding])
    function_name_embeddings = model.encode(function_name_list)
    function_name_embeddings_list=[]
    for embedding in function_name_embeddings:
        function_name_embeddings_list.append([embedding])

    function_param_embeddings = model.encode(function_param_list)
    function_param_embeddings_list=[]
    for embedding in function_param_embeddings:
        function_param_embeddings_list.append([embedding])

    function_body_embeddings = model.encode(function_body_list)
    function_body_embeddings_list=[]
    for embedding in function_body_embeddings:
        function_body_embeddings_list.append([embedding])


    #func_dict={'Directory':dir_name_list,'Filename':filename_list,'Class Name':class_name_list,'Class Name embeddings':class_name_embeddings_list,'Parent Class':parent_class_list,'Parent Class embeddings':parent_class_embeddings_list,'Function Name':function_name_list,'Function Name embeddings':function_name_embeddings_list,'Function Parameters':function_param_list,'Function Parameter embeddings':function_param_embeddings_list,'Function Body':function_body_list,'Function Body embeddings':function_body_embeddings_list}
    func_dict={'Directory':dir_name_list,'Filename':filename_list,'Class Name':class_name_list,'Class Name embeddings':class_name_embeddings_list,'Parent Class':parent_class_list,'Function Name':function_name_list,'Function Name embeddings':function_name_embeddings_list,'Function Parameters':function_param_list,'Function Parameter embeddings':function_param_embeddings_list,'Function Body':function_body_list,'Function Body embeddings':function_body_embeddings_list}

    func_df=pd.DataFrame(func_dict)
    func_df_final = pd.concat([func_df_final, func_df], ignore_index=True)
func_df_final.to_pickle('func_dataframe.pkl')
func_df_final.to_excel('func_dataframe.xlsx',index=False)


