package com.visiongoogle;

import com.google.api.gax.longrunning.OperationFuture;
import com.google.cloud.storage.Blob;
import com.google.cloud.storage.Bucket;
import com.google.cloud.storage.Storage;
import com.google.cloud.storage.StorageOptions;
import com.google.cloud.vision.v1.*;
import com.google.protobuf.ByteString;
import com.google.protobuf.util.JsonFormat;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Vision {

    // [START vision_logo_detection]
    public static void detectLogos(String filePath, PrintStream out) throws Exception, IOException {
        List<AnnotateImageRequest> requests = new ArrayList<>();

        ByteString imgBytes = ByteString.readFrom(new FileInputStream(filePath));

        Image img = Image.newBuilder().setContent(imgBytes).build();
        Feature feat = Feature.newBuilder().setType(Feature.Type.LOGO_DETECTION).build();
        AnnotateImageRequest request =
                AnnotateImageRequest.newBuilder().addFeatures(feat).setImage(img).build();
        requests.add(request);

        try (ImageAnnotatorClient client = ImageAnnotatorClient.create()) {
            BatchAnnotateImagesResponse response = client.batchAnnotateImages(requests);
            List<AnnotateImageResponse> responses = response.getResponsesList();

            for (AnnotateImageResponse res : responses) {
                if (res.hasError()) {
                    out.printf("Error: %s\n", res.getError().getMessage());
                    return;
                }

                // For full list of available annotations, see http://g.co/cloud/vision/docs
                for (EntityAnnotation annotation : res.getLogoAnnotationsList()) {
                    out.println(annotation.getDescription());
                }
            }
        }
    }

    public static void detectText(String filePath, PrintStream out) throws Exception, IOException {
        List<AnnotateImageRequest> requests = new ArrayList<>();

        ByteString imgBytes = ByteString.readFrom(new FileInputStream(filePath));

        Image img = Image.newBuilder().setContent(imgBytes).build();
        Feature feat = Feature.newBuilder().setType(Feature.Type.TEXT_DETECTION).build();
        AnnotateImageRequest request =
                AnnotateImageRequest.newBuilder().addFeatures(feat).setImage(img).build();
        requests.add(request);

        try (ImageAnnotatorClient client = ImageAnnotatorClient.create()) {
            BatchAnnotateImagesResponse response = client.batchAnnotateImages(requests);
            List<AnnotateImageResponse> responses = response.getResponsesList();

            for (AnnotateImageResponse res : responses) {
                if (res.hasError()) {
                    out.printf("Error: %s\n", res.getError().getMessage());
                    return;
                }

                // For full list of available annotations, see http://g.co/cloud/vision/docs
                for (EntityAnnotation annotation : res.getTextAnnotationsList()) {
                    out.printf("Text: %s\n", annotation.getDescription());
                    out.printf("Position : %s\n", annotation.getBoundingPoly());
                }
            }
        }
    }


    public static void detectProperties(String filePath, PrintStream out) throws Exception,
            IOException {
        List<AnnotateImageRequest> requests = new ArrayList<>();

        ByteString imgBytes = ByteString.readFrom(new FileInputStream(filePath));

        Image img = Image.newBuilder().setContent(imgBytes).build();
        Feature feat = Feature.newBuilder().setType(Feature.Type.IMAGE_PROPERTIES).build();
        AnnotateImageRequest request =
                AnnotateImageRequest.newBuilder().addFeatures(feat).setImage(img).build();
        requests.add(request);

        try (ImageAnnotatorClient client = ImageAnnotatorClient.create()) {
            BatchAnnotateImagesResponse response = client.batchAnnotateImages(requests);
            List<AnnotateImageResponse> responses = response.getResponsesList();

            for (AnnotateImageResponse res : responses) {
                if (res.hasError()) {
                    out.printf("Error: %s\n", res.getError().getMessage());
                    return;
                }

                // For full list of available annotations, see http://g.co/cloud/vision/docs
                DominantColorsAnnotation colors = res.getImagePropertiesAnnotation().getDominantColors();
                for (ColorInfo color : colors.getColorsList()) {
                    out.printf(
                            "fraction: %f\nr: %f, g: %f, b: %f\n",
                            color.getPixelFraction(),
                            color.getColor().getRed(),
                            color.getColor().getGreen(),
                            color.getColor().getBlue());
                }
            }
        }
    }

    public static void detectDocumentsGcs(String gcsSourcePath, String gcsDestinationPath) throws Exception {

        try (ImageAnnotatorClient client = ImageAnnotatorClient.create()) {
            List<AsyncAnnotateFileRequest> requests = new ArrayList<>();

            // Set the GCS source path for the remote file.
            GcsSource gcsSource = GcsSource.newBuilder()
                    .setUri(gcsSourcePath)
                    .build();

            InputConfig inputConfig = InputConfig.newBuilder()
                    .setMimeType("application/pdf") // Supported MimeTypes: "application/pdf", "image/tiff"
                    .setGcsSource(gcsSource)
                    .build();

            // Set the GCS destination path for where to save the results.
            GcsDestination gcsDestination = GcsDestination.newBuilder()
                    .setUri(gcsDestinationPath)
                    .build();

            OutputConfig outputConfig = OutputConfig.newBuilder()
                    .setBatchSize(2)
                    .setGcsDestination(gcsDestination)
                    .build();

            // Select the Feature required by the vision API
            Feature feature = Feature.newBuilder().setType(Feature.Type.DOCUMENT_TEXT_DETECTION).build();

            // Build the OCR request
            AsyncAnnotateFileRequest request = AsyncAnnotateFileRequest.newBuilder()
                    .addFeatures(feature)
                    .setInputConfig(inputConfig)
                    .setOutputConfig(outputConfig)
                    .build();
            requests.add(request);

            // Perform the OCR request
            OperationFuture<AsyncBatchAnnotateFilesResponse, OperationMetadata> response =
                    client.asyncBatchAnnotateFilesAsync(requests);

            System.out.println("Waiting for the operation to finish.");

            // Wait for the request to finish. (The result is not used, since the API saves the result to
            // the specified location on GCS.)
            List<AsyncAnnotateFileResponse> result = response.get(180, TimeUnit.SECONDS)
                    .getResponsesList();
            Storage storage = StorageOptions.getDefaultInstance().getService();

            // Get the destination location from the gcsDestinationPath
            Pattern pattern = Pattern.compile("gs://([^/]+)/(.+)");
            Matcher matcher = pattern.matcher(gcsDestinationPath);

            if (matcher.find()) {
                String bucketName = matcher.group(1);
                String prefix = matcher.group(2);

                // Get the list of objects with the given prefix from the GCS bucket
                Bucket bucket = storage.get(bucketName);
                com.google.api.gax.paging.Page<Blob> pageList = bucket.list(Storage.BlobListOption.prefix(prefix));

                Blob firstOutputFile = null;

                // List objects with the given prefix.
                System.out.println("Output files:");
                for (Blob blob : pageList.iterateAll()) {
                    System.out.println(blob.getName());
                    if (firstOutputFile == null) {
                        firstOutputFile = blob;
                    }
                }
                String jsonContents = new String(firstOutputFile.getContent());
                AnnotateFileResponse.Builder builder = AnnotateFileResponse.newBuilder();
                JsonFormat.parser().merge(jsonContents, builder);

                // Build the AnnotateFileResponse object
                AnnotateFileResponse annotateFileResponse = builder.build();

                // Parse through the object to get the actual response for the first page of the input file.
                AnnotateImageResponse annotateImageResponse = annotateFileResponse.getResponses(0);
                System.out.format("\nText: %s\n", annotateImageResponse.getFullTextAnnotation().getText());
            } else {
                System.out.println("No MATCH");
            }
        }
    }


}
