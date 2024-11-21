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
	maxRowConcurrency   = 5 // Maximum number of rows (batches) processed concurrently
	arrowFilePath       = "../../obelics-train-converted.arrow"
	outputArrowFilePath = "processed_images.arrow"
	downloadTimeout     = 10 * time.Second
	maxGoroutines       = 10 // Limit the number of concurrent goroutines

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

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed to fetch image: %s", resp.Status)
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body for %s: %v", url, err)
	}

	return data, nil
}

// DownloadImages downloads all images for a single row without limiting concurrency.
func DownloadImages(urls []string, client *http.Client, mu2 *sync.Mutex) {
	// var wg sync.WaitGroup
	// images := make([][]byte, len(urls)) // Initialize with nils

	for i, url := range urls {
		if url == "" {
			continue // Skip None-equivalent URLs but keep index for order
		}
		fmt.Println(i)
		// wg.Add(1)
		func(i int, url string) {
			// defer wg.Done()
			// data, err := FetchImage(url, client)
			// if err != nil {
			// 	log.Printf("\n\n\n\n")
			// 	log.Printf("Error fetching image at %s: %v\n", url, err)
			// 	log.Printf("\n\n\n\n")
			// 	log.Printf("yaha to aaya1")
			// 	// mu2.Lock()
			// 	images[i] = nil
			// 	// mu2.Unlock()
			// 	log.Printf("yaha to aaya2")
			// } else {
			// 	// mu2.Lock()
			// 	images[i] = data // Place the image data at the correct index
			// 	// mu2.Unlock()
			// }
		}(i, url)
	}

	// wg.Wait()
	// return images
}

// ProcessRow handles downloading images for one row (batch) concurrently.
func ProcessRow(record arrow.Record, client *http.Client, allImageData *[][]byte, mu *sync.Mutex) {
	// Access the first column, assuming it represents a list
	// var mu2 sync.Mutex
	col := record.Column(0)
	if col == nil {
		log.Fatalf("Column is nil")
	}
	if listCol, ok := col.(*array.List); ok {
		fmt.Println("Column type is *array.List")
		fmt.Println("Number of rows:", listCol.Len())

		// Flattened list of all values and offsets for reconstructing sublists
		urlsArray := listCol.ListValues().(*array.String)
		offsets := listCol.Offsets()

		if urlsArray == nil || urlsArray.Len() == 0 {
			log.Fatalf("urlsArray is nil or empty")
		}

		// Iterate over each row in the column
		for i := 0; i < listCol.Len(); i++ {
			if i+1 >= len(offsets) {
				log.Fatalf("Offset index out of bounds: i=%d, len(offsets)=%d", i, len(offsets))
			}
			fmt.Println(i, "this is indie url")
			if offsets[i+1] <= offsets[i] {
				log.Fatalf("Invalid offset range: start=%d, end=%d", offsets[i], offsets[i+1])
			}

			start, end := offsets[i], offsets[i+1]
			urls := make([]string, end-start)

			// Extract URLs for the current row
			for j := start; j < end; j++ {
				urls[j-start] = urlsArray.Value(int(j))
			}

			// Download images for this list of URLs
			// _ = DownloadImages(urls, client, &mu2)

			// // Append downloaded images to the shared data structure
			// mu.Lock()
			// *allImageData = append(*allImageData, images...)
			// mu.Unlock()
		}
	} else {
		log.Fatalf("Column 0 is not a *array.List")
	}
}

// ProcessArrowFile loads the Arrow file, processes rows sequentially, and saves results to a new Arrow file.
func ProcessArrowFile() {
	file, err := os.Open(arrowFilePath)
	if err != nil {
		log.Fatalf("Failed to open Arrow file: %v", err)
	}
	defer file.Close()

	reader, err := ipc.NewFileReader(file, ipc.WithAllocator(memory.NewGoAllocator()))
	if err != nil {
		log.Fatalf("Failed to create Arrow reader: %v", err)
	}
	defer reader.Close()

	client := &http.Client{}
	var allImageData [][]byte
	var mu sync.Mutex
	var wg sync.WaitGroup
	semaphore := make(chan struct{}, maxGoroutines) // Buffered channel for limiting concurrency

	for i := 0; i < reader.NumRecords(); i++ {
		record, err := reader.Record(i)
		if err != nil {
			log.Fatalf("Failed to read record: %v", err)
		}

		// Call ProcessRow sequentially
		wg.Add(1)
		semaphore <- struct{}{}
		go func(rec arrow.Record) {
			defer func() { <-semaphore }()
			defer wg.Done()
			ProcessRow(rec, client, &allImageData, &mu)
		}(record)
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
