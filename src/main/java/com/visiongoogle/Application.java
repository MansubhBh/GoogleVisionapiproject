package com.visiongoogle;

public class Application {

    public static void main(String[] args) {

        System.out.println("Trying the google cloud vision api");
        try {
            Vision.detectDocumentsGcs("gs://simple-pdf-soruce/pdf-test.pdf","gs://simple-pdf-destination/response/");
            //Vision.detectText("/Users/mansubh/projects/visiongoogle/simple.jpg", new PrintStream("/Users/mansubh/projects/visiongoogle/result1.txt"));

        }catch (Exception e){
            System.out.println("Exception -> "+ e.getMessage());
        }
    }





}
