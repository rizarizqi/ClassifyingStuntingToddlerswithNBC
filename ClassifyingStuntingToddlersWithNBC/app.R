#Created by: Riza Rizqi Robbi Arisandi


library(shiny)
library(NLP)
library(xml2)
library(plyr)
library(tidyverse)
library(tidytext)
library(dplyr)
library(stringr)
library(data.table)
library(cvTools)
library(caret)
library(RColorBrewer)
library(e1071)
library(remotes)
library(shiny)
library(shinythemes)
library(shinyWidgets)
library(shinycssloaders)
library(DT)
library(markdown)
library(sass)
library(class)

# Define UI for application that draws a histogram
ui <- fluidPage(theme = shinytheme("superhero"),
                #Judul Aplikasi
                titlePanel("Klasifikasi Status Gizi Stunting"),
                h4("Naïve Bayes Classifier (NBC) dan K-Fold Cross Vaidation"),
                navbarPage("App",
                           #Tab Home
                           tabPanel("Home",
                                    br(),
                                    div(
                                        h3("Aplikasi Naïve Bayes Classifier (NBC)
                                           Pada Klasifikasi Status Gizi Balita 
                                           Stunting Dengan Pengujian K-Fold Cross Validation",
                                           style="text-align:center")
                                    ),
                                    br(), 
                                    br(),
                                    h5("Created by:", style="text-align:center"),
                                    h5("Riza Rizqi Robbi Arisandi", 
                                       style="text-align:center"),
                                    br(), 
                                    br(),
                                    h4("Departemen Statistika", 
                                       style="text-align:center"),
                                    h4("Fakultas Sains dan Matematika", 
                                       style="text-align:center"),
                                    h4("Universitas Diponegoro", 
                                       style="text-align:center"),
                                    h4("2022", style="text-align:center")
                           ),
                           
                           #Tab Dataset
                           tabPanel("Input Data",
                                    sidebarLayout(
                                        sidebarPanel(
                                            fileInput("data","Choose CSV File",
                                                      multiple = FALSE,
                                                      accept = c("text/csv",
                                                                 "text/comma-separated-values,
                                                                 text/plain",
                                                                 ".csv")),
                                            #Horizontal line
                                            tags$hr(),
                                            
                                            #Input Checkbox if file has header
                                            checkboxInput("header",
                                                          "Header",
                                                          TRUE),
                                            
                                            #Input: select separator
                                            radioButtons("sep", "Separator",
                                                         choices = c(Comma=",",
                                                                     semicolon=";",
                                                                     Tab="\t"),
                                                         selected=";"),
                                            
                                            #Input: select quotes
                                            radioButtons("quote", "Quote",
                                                         choices = c(None="",
                                                                     "Double Quote"='"',
                                                                     "Single Quote"="'"),
                                                         selected = '"'),                                           
                                            #Horizontal line
                                            tags$hr(),                                            
                                            #Input: select number of rows to display
                                            radioButtons("disp",
                                                         "Display",
                                                         choices = c(Head = "head",
                                                                     All = "all"),
                                                         selected = "head")
                                        ),
                                        mainPanel(DTOutput("data")),
                                        position="left"
                                    )),                           
                           
                           #Tab Summary
                           tabPanel("Statistik Deskriptif",
                                    verbatimTextOutput("summ")),
                           
                           #NBC
                           tabPanel("Naive Bayes Classifier",
                                    sidebarLayout(
                                        sidebarPanel(
                                            numericInput('set.seed', 
                                                         'Input random seed:'
                                                         ,NULL),
                                            hr(),
                                            numericInput('K1', 
                                                         'Input K for K-Fold Cross Validation:',
                                                         10),
                                            actionButton("run1","RUN")
                                        ),
                                        mainPanel(style = "color:white",
                                                  h4("Hasil Klasifikasi", 
                                                     style="text-align:left"),
                                                  verbatimTextOutput("klasifikasi.nbc"),
                                                  hr(),
                                                  
                                        ),                                    
                                    )
                           ), 
                           
                           #Kinerja Klasifikasi
                           tabPanel("Kinerja Klasifikasi",
                                    sidebarLayout(
                                        sidebarPanel(
                                            h4("Akurasi", 
                                               style="text-align:left"),
                                            actionButton("run2",
                                                         "PRINT"),
                                            hr(),
                                            h4("Cetak Matriks Konfusi", 
                                               style="text-align:left"),
                                            actionButton("run3","PRINT")
                                        ),
                                        mainPanel(style = "color:white",
                                                  h4("Hasil Kinerja Klasifikasi", 
                                                     style="text-align:left"),
                                                  verbatimTextOutput("hasil.nbc"),
                                                  hr(),
                                                  h4("Matriks Konfusi Setiap Fold:", 
                                                     style="text-align:left"),
                                                  verbatimTextOutput("conf.mat")
                                        ),                                    
                                    )
                           ), 
                           
                           #Tab Prediksi
                           tabPanel("Prediksi",
                                    sidebarLayout(
                                        sidebarPanel(
                                            h4("Input Data", 
                                               style="text-align:left"),
                                            fileInput("file2","Choose CSV File",
                                                      multiple = FALSE,
                                                      accept = c("text/csv",
                                                                 "text/comma-separated-values,
                                                                 text/plain",
                                                                 ".csv")),
                                            
                                            
                                            #Input Checkbox if file has header
                                            checkboxInput("header",
                                                          "Header",
                                                          TRUE),
                                            
                                            #Input: select separator
                                            radioButtons("sep", "Separator",
                                                         choices = c(Comma=",",
                                                                     semicolon=";",
                                                                     Tab="\t"),
                                                         selected=";"),
                                            
                                            #Input: select quotes
                                            radioButtons("quote", "Quote",
                                                         choices = c(None="",
                                                                     "Double Quote"='"',
                                                                     "Single Quote"="'"),
                                                         selected = '"'),                                           
                                            
                                            #Input: select number of rows to display
                                            radioButtons("disp",
                                                         "Display",
                                                         choices = c(Head = "head",
                                                                     All = "all"),
                                                         selected = "head"),
                                            
                                            #Horizontal line
                                            tags$hr(), 
                                            
                                            h4("Pilih Data Train", 
                                               style="text-align:left"),
                                            radioButtons("iter", "Fold Ke:",
                                                         choices = c("Fold 1"=1,
                                                                     "Fold 2"=2,
                                                                     "Fold 3"=3,
                                                                     "Fold 4"=4,
                                                                     "Fold 5"=5,
                                                                     "Fold 6"=6,
                                                                     "Fold 7"=7,
                                                                     "Fold 8"=8,
                                                                     "Fold 9"=9,
                                                                     "Fold 10"=10),
                                                         selected = NULL),
                                            actionButton("run4","RUN")
                                            
                                        ),
                                        mainPanel(style = "color:white",
                                                  h4("Data", 
                                                     style="text-align:left"),
                                                  DTOutput("contents2"),
                                                  hr(),
                                                  h4("Hasil Prediksi", 
                                                     style="text-align:left"),
                                                  verbatimTextOutput("prediksi.nbc"),
                                                  
                                        )
                                    ))
                           
                )
)

server <- function(input, output, session){
    #Input Data
    data.input <- reactive({
        in.file <- input$data
        if(is.null(in.file)){
            return(NULL)
        }else
            df <- read.csv(input$data$datapath,
                           header = input$header,
                           sep = input$sep,
                           quote = input$quote)
    })
    #Print Data
    output$data = renderDT({
        data = data.input()
    }) 
    
    #Tab Summary
    output$summ = renderPrint({
        req(input$data)
        dataku <- read.csv(input$data$datapath,
                           header = input$header,
                           sep = input$sep,
                           quote = input$quote)
        
        #Mengubah jenis data sesuai dengan jenisnya
        dataku["STATUS"]=lapply(dataku["STATUS"],factor)
        dataku["JK"]=lapply(dataku["JK"],factor)
        dataku["UMUR"]=lapply(dataku["UMUR"],as.numeric)
        dataku["LILA"]=lapply(dataku["LILA"],as.numeric)
        summary(dataku)
    })
    
    #Tab Naive Bayes Classifier
    observeEvent(input$run1,{
        withProgress(message = 'Naive Bayes Classifier in Progress',
                     detail = 'Please wait ...', {
                         
                         #Nilai K untuk K-Fold Cross Validation                          
                         K = input$K1
                         req(input$data)
                         dataku <- read.csv(input$data$datapath,
                                            header = input$header,
                                            sep = input$sep,
                                            quote = input$quote)
                         dataku["STATUS"]=lapply(dataku["STATUS"],factor)
                         dataku["JK"]=lapply(dataku["JK"],factor)
                         dataku["UMUR"]=lapply(dataku["UMUR"],as.numeric)
                         dataku["LILA"]=lapply(dataku["LILA"],as.numeric)
                         
                         #Input random seed sebagai angka acalk
                         set.seed(input$set.seed)
                         
                         #Pembagian Data latih dan Data Uji 
                         rows <- sample(nrow(dataku))
                         dataku <- dataku[rows, ]
                         
                         #Confusion Matrix
                         NBC.i=function(dataku,K,i)
                         {
                             folds <- rep_len(1:K, nrow(dataku))
                             Akurasi = c(rep(0,K))
                             
                             # actual split of the data
                             fold <- which(folds == i)
                             data.train <- dataku[-fold,]
                             data.test <- dataku[fold,]
                             classifier<- naiveBayes(STATUS~.,
                                                     data=data.train)
                             pred<-predict(classifier, data.test)
                             hasil=cbind(data.test,pred)
                             colnames(hasil)= c("USIA",
                                                "JK",
                                                "TB",
                                                "BB",
                                                "LiLA",
                                                "STATUS AKTUAL",
                                                "STATUS PREDIKSI")
                             cat("Hasil Klasifikasi Fold ke:",i,"\n")
                             cat("-------------------------------------------------------------------\n")
                             print(hasil)
                             cat("-------------------------------------------------------------------\n")
                         }
                         
                         output$klasifikasi.nbc<-renderPrint({
                             for(i in 1:K){
                                 NBC.i(dataku, K,i)
                             }
                         })
                         
                         
                     }
        )
    }
    )
    
    
    #Kinerja
    observeEvent(input$run2,{
        withProgress(message = 'Naive Bayes Classifier in Progress',
                     detail = 'Please wait ...', {
                         
                         #Nilai K untuk K-Fold Cross Validation                          
                         K = input$K1
                         req(input$data)
                         dataku <- read.csv(input$data$datapath,
                                            header = input$header,
                                            sep = input$sep,
                                            quote = input$quote)
                         dataku["STATUS"]=lapply(dataku["STATUS"],factor)
                         dataku["JK"]=lapply(dataku["JK"],factor)
                         dataku["UMUR"]=lapply(dataku["UMUR"],as.numeric)
                         dataku["LILA"]=lapply(dataku["LILA"],as.numeric)
                         
                         #Input random seed sebagai angka acalk
                         set.seed(input$set.seed)
                         
                         #Pembagian Data latih dan Data Uji 
                         rows <- sample(nrow(dataku))
                         dataku <- dataku[rows, ]
                         
                         incProgress(1/3) #Penanda progress loading
                         
                         NBC=function(dataku,K){
                             folds <- rep_len(1:K, nrow(dataku))
                             Akurasi = c(rep(0,K))
                             
                             for(i in 1:K){
                                 #actual split of the data
                                 fold <- which(folds == i)
                                 data.train <- dataku[-fold,]
                                 data.test <- dataku[fold,]
                                 
                                 #Klasifikasi
                                 classifier<- naiveBayes(STATUS~.,
                                                         data=data.train)
                                 pred<-predict(classifier,
                                               data.test)
                                 
                                 #Hasil Klasifikasi
                                 xtab<-table("Predictions"= pred,
                                             "Actual"= data.test[,"STATUS"])
                                 conf.mat<-confusionMatrix(xtab,
                                                           mode="prec_recall")
                                 Akurasi[i] = conf.mat$overall[1]
                                 
                             }
                             return(Akurasi)
                             
                         }
                         incProgress(2/3) #Penanda progress loading
                         
                         #menampilkan hasil klasifikasi
                         akurasi.nbc = NBC(dataku, K)
                         fold.i = c(1:K)
                         Akurasi.Tabel = cbind(fold.i,akurasi.nbc)
                         colnames(Akurasi.Tabel)= c("Fold ke-",
                                                    "Akurasi")
                         
                         output$hasil.nbc<-renderPrint({
                             Akurasi.Tabel
                         })
                         #Fungsi NBC
                         
                         
                         incProgress(3/3) #Penanda progress loading
                         
                         #Confusion Matrix
                         NBC.i=function(dataku,K,i)
                         {
                             folds <- rep_len(1:K, nrow(dataku))
                             Akurasi = c(rep(0,K))
                             
                             # actual split of the data
                             fold <- which(folds == i)
                             data.train <- dataku[-fold,]
                             data.test <- dataku[fold,]
                             classifier<- naiveBayes(STATUS~.,
                                                     data=data.train)
                             pred<-predict(classifier, data.test)
                             xtab<-table("Predictions"= pred,
                                         "Actual"= data.test[,"STATUS"])
                             conf.mat<-confusionMatrix(xtab,
                                                       mode="prec_recall")
                             hasil=cbind(data.test,pred)
                             colnames(hasil)= c("USIA",
                                                "JK",
                                                "TB",
                                                "BB",
                                                "LiLA",
                                                "STATUS AKTUAL",
                                                "STATUS PREDIKSI")
                             
                             cat("Matriks Konfusi NBC Fold ke:",i,"\n")
                             cat("-------------------------------------------------------------------\n")
                             print(conf.mat)
                             cat("-------------------------------------------------------------------\n")
                         }
                         observeEvent(input$run3 ,{
                             output$conf.mat<-renderPrint({
                                 for(i in 1:K){
                                     NBC.i(dataku, K,i)
                                 }
                             })
                         })
                         
                     })
    })
    
    #Tab Prediksi
    #data
    output$contents2 = renderDT({
        req(input$file2)
        df2 <- read.csv(input$file2$datapath,
                        header = input$header,
                        sep = input$sep,
                        quote = input$quote)
        return(df2)
    })
    
    
    #Model Prediksi
    observeEvent(input$run4,{
        req(input$data)
        dataku <- read.csv(input$data$datapath,
                           header = input$header,
                           sep = input$sep,
                           quote = input$quote)
        dataku["STATUS"]=lapply(dataku["STATUS"],factor)
        dataku["JK"]=lapply(dataku["JK"],factor)
        dataku["UMUR"]=lapply(dataku["UMUR"],as.numeric)
        dataku["LILA"]=lapply(dataku["LILA"],as.numeric)
        
        req(input$file2)
        data.pred <- read.csv(input$file2$datapath,
                              header = input$header,
                              sep = input$sep,
                              quote = input$quote)
        data.pred["JK"]=lapply(data.pred["JK"],factor)
        data.pred["UMUR"]=lapply(data.pred["UMUR"],as.numeric)
        data.pred["LILA"]=lapply(data.pred["LILA"],as.numeric)
        
        req(input$K1)
        K=input$K1
        
        #Pembagian Data latih dan Data Uji 
        rows <- sample(nrow(dataku))
        dataku <- dataku[rows, ]
        
        pred.NBC=function(dataku,K){
            folds <- rep_len(1:K, nrow(dataku))
            i = input$iter
            #actual split of the data
            fold <- which(folds == i)
            data.train <- dataku[-fold,]
            data.test <- data.pred
            
            #Klasifikasi
            classifier<- naiveBayes(STATUS~.,
                                    data=data.train)
            pred<-predict(classifier,
                          data.test)
            
            #Hasil Klasifikasi
            hasil.pred=cbind(data.test,pred)
            colnames(hasil.pred)= c("USIA",
                               "JK",
                               "TB",
                               "BB",
                               "LiLA",
                               "STATUS PREDIKSI")
            cat("-------------------------------------------------------------------\n")
            print(hasil.pred)
            cat("-------------------------------------------------------------------\n")
        }
        output$prediksi.nbc<-renderPrint({
            pred.NBC(dataku, K)
        })
    })
}

# Run the application 
shinyApp(ui = ui, server = server)
