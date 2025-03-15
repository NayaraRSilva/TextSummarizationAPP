# IMPORTANDO AS BIBLIOTECAS
from langchain_anthropic import ChatAnthropic                   # Importando a biblioteca anthropic
from langchain.docstore.document import Document                # Importando a biblioteca Document permite a criação de documentos
from langchain.text_splitter import CharacterTextSplitter       # Importando a biblioteca CharacterTextSplitter que permite a divisão de texto
from langchain.chains.summarize import load_summarize_chain     # Importando a biblioteca load_summarize_chain que permite a sumarização de texto
from dotenv import load_dotenv, find_dotenv                     # Importando a biblioteca dotenv que permite a leitura de variáveis de ambiente
import os                                                       # Importando a biblioteca os que permite a manipulação de variáveis de ambiente

# CARREGANDO AS VARIÁVEIS DE AMBIENTE
load_dotenv(find_dotenv())
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

# CRIAR MODELO AI
llm = ChatAnthropic(
    model = "claude-3-opus-20240229",
    temperature= 0, # Ajusta o nível de criatividade do modelo
    anthropic_api_key = ANTHROPIC_API_KEY


)

text = "Mary del Priore é autora de 54 livros de História do Brasil e ganhadora de mais de 20 prêmios literários, nacionais e internacionais entre os quais, quatro Jabutis. Lecionou nos Departamentos de História da FFLCH/USP, PUC/RJ e Universidade Salgado de Oliveira. Além de sócia titular do Instituto Histórico e Geográfico Brasileiro, pertence a mais de 11 academias de história nacionais e internacionais. Sócia da Academia Carioca de Letras, da Academia Paulista de Letras e do PEN Club do Brasil é membro do Conselho de Notáveis da Confederação Nacional de Comércio (CNC) e da Comissão Científica Internacional da Cátedra Infante Dom Henrique para Estudos Insulares Atlânticos e Globalização (CIDH) sediada na Universidade Aberta de Lisboa. Membro do Conselho do Patrimônio Cultural da Santa Casa de Misericórdia do Rio de Janeiro, é também autora brasileira selecionada para o catálogo da Câmara Brasileira do Livro (CBL). Como consultora de consagrados diretores de cinema, contribui com conteúdo históricos para filmes e documentários realizados para a televisão brasileira. Foi eleita em 2021, pela opinião pública através da revista “Aventuras na História” a “Melhor historiadora do Brasil” e “Escritora de Destaque”."

# SPLIT TEXT : Fatiamento do Texto
text_splitter = CharacterTextSplitter()
texts = text_splitter.split_text(text)

# CREATE DOCUMENTS : CRIAÇÃO DE DOCUMENTOS

docs = [Document(page_content=text) for text in texts]

#sumarização dos textos 
chain = load_summarize_chain(llm=llm, chain_type='map_reduce')

# excutar chain
summary = chain.invoke(docs)
print(summary['output_text'])