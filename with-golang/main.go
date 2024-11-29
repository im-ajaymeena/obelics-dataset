package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/apache/arrow/go/v12/arrow"
	"github.com/apache/arrow/go/v12/arrow/array"
	"github.com/apache/arrow/go/v12/arrow/ipc"
	"github.com/apache/arrow/go/v12/arrow/memory"
)

const (
	maxRowConcurrency   = 10 // Maximum number of rows (batches) processed concurrently
	arrowFilePath       = "../obelics-train-converted.arrow"
	outputArrowFilePath = "processed_images.arrow"
	downloadTimeout     = 10 * time.Second
)

// FetchImage downloads an image from the provided URL.
func FetchImage(url string, client *http.Client) ([]byte, error) {
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}

	ctx, cancel := context.WithTimeout(context.Background(), downloadTimeout)
	defer cancel()

	req = req.WithContext(ctx)
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	// if resp.StatusCode != http.StatusOK {
	// 	return nil, fmt.Errorf("failed to fetch image: %s", resp.Status)
	// }

	data, err := io.ReadAll(resp.Body)
	// if err != nil {
	// 	return nil, fmt.Errorf("failed to read response body for %s: %v", url, err)
	// }

	return data, nil
}

// DownloadImages downloads all images for a single row without limiting concurrency.
func DownloadImages(urls []string, client *http.Client) [][]byte {
	var wg sync.WaitGroup
	images := make([][]byte, len(urls)) // Initialize with nils

	for i, url := range urls {
		if url == "" {
			continue // Skip None-equivalent URLs but keep index for order
		}
		wg.Add(1)
		go func(i int, url string) {
			defer wg.Done()
			data, err := FetchImage(url, client)
			if err != nil {
				// log.Printf("\n\n\n\n")
				// log.Printf("Error fetching image at %s: %v\n", url, err)
				// log.Printf("\n\n\n\n")
				// log.Printf("yaha to aaya1")
				images[i] = nil
				// log.Printf("yaha to aaya2")
			} else {
				images[i] = data // Place the image data at the correct index
			}
		}(i, url)
	}

	wg.Wait()
	return images
}

// ProcessRow handles downloading images for one row (batch) concurrently.
func ProcessRow(record arrow.Record, client *http.Client, rowSemaphore chan int, wg *sync.WaitGroup, allImageData *[][]byte) {
	defer wg.Done()
	defer func() { <-rowSemaphore }() // Release row semaphore when done

	// Treat the "images" column as a list
	imagesCol := record.Column(0).(*array.List)
	urlsArray := imagesCol.ListValues().(*array.String) // Access list values as strings
	offsets := imagesCol.Offsets()                      // Offset array for list elements

	// Iterate over each row in the list
	for i := 0; i < imagesCol.Len(); i++ {
		start, end := offsets[i], offsets[i+1]
		urls := make([]string, end-start)
		for j := start; j < end; j++ {
			urls[j-start] = urlsArray.Value(int(j)) // Extract each URL string
		}

		// Print URLs for the current row
		// fmt.Printf("Row %d URLs: %v\n", i, urls)

		// Download images for this list of URLs
		images := DownloadImages(urls, client)
		*allImageData = append(*allImageData, images...)
		// log.Printf("yaha to aaya3")
	}
}

// ProcessArrowFile loads the Arrow file, processes multiple rows concurrently, and saves results to a new Arrow file.
func ProcessArrowFile() {
	file, err := os.Open(arrowFilePath)
	if err != nil {
		log.Fatalf("Failed to open Arrow file: %v", err)
	}
	defer file.Close()

	reader, err := ipc.NewFileReader(file, ipc.WithAllocator(memory.NewGoAllocator()))
	// if err != nil {
	// 	log.Fatalf("Failed to create Arrow reader: %v", err)
	// }
	defer reader.Close()

	client := &http.Client{}
	rowSemaphore := make(chan int, maxRowConcurrency) // Control row concurrency

	var wg sync.WaitGroup
	var allImageData [][]byte
	numRecords := reader.NumRecords()

	for i := 0; i < numRecords; i++ {
		record, err := reader.Record(i)
		if err != nil {
			log.Fatalf("Failed to read record: %v", err)
		}

		rowSemaphore <- 1 // Acquire row semaphore
		wg.Add(1)
		go ProcessRow(record, client, rowSemaphore, &wg, &allImageData)
	}

	wg.Wait()

	// Write the downloaded images back to a new Arrow file
	allocator := memory.NewGoAllocator()
	writeToArrowFile(allImageData, allocator)
}

// writeToArrowFile writes downloaded images to an Arrow file in binary format.
func writeToArrowFile(imageData [][]byte, allocator memory.Allocator) {
	// Define Arrow schema with a binary column for images
	fields := []arrow.Field{
		{Name: "images", Type: arrow.BinaryTypes.Binary},
	}
	schema := arrow.NewSchema(fields, nil)

	// Build the Arrow record batch with image data
	builder := array.NewBinaryBuilder(allocator, arrow.BinaryTypes.Binary)
	defer builder.Release()

	for _, data := range imageData {
		builder.Append(data)
	}

	records := array.NewRecord(schema, []arrow.Array{builder.NewArray()}, int64(len(imageData)))
	defer records.Release()

	// Write the record to a new Arrow file
	outputFile, err := os.Create(outputArrowFilePath)
	if err != nil {
		log.Fatalf("Failed to create Arrow output file: %v", err)
	}
	defer outputFile.Close()

	writer, err := ipc.NewFileWriter(outputFile, ipc.WithSchema(schema))
	defer writer.Close()

	if err := writer.Write(records); err != nil {
		log.Fatalf("Failed to write record to Arrow file: %v", err)
	}

	fmt.Println("Successfully saved images to Arrow file:", outputArrowFilePath)
}

func main() {
	ProcessArrowFile()
}
