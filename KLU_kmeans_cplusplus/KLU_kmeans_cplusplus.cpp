#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <stdexcept>
#include <random>
#include  <iterator>

//Constants for file locations,
//and for k-means settings
const std::string DATA_FOLDER = "data/";
const std::string OUTPUTS_FOLDER = "outputs/";
const std::string DATA_FILENAME = DATA_FOLDER + "unbalance.txt";
const std::string CENTROID_FILENAME = OUTPUTS_FOLDER + "centroid.txt";
const std::string PARTITION_FILENAME = OUTPUTS_FOLDER + "partition.txt";
const int NUM_CENTROIDS = 8;  // Number of clusters
const char SEPARATOR = ' ';
const int MAX_ITERATIONS = 25;
const int MAX_REPEATS = 5;


// Class representing a data point
class DataPoint {
public:
    // Vector to hold the attributes of the data point
    std::vector<int> attributes;

    // Minimum distance from this data point to a centroid
    double minDistance = std::numeric_limits<double>::max();

    // Default constructor
	DataPoint() {};

    // Constructor to initialize the data point with given attributes
	explicit DataPoint(const std::vector<int>& attributes) : attributes(attributes) {};
	
	// Overload equality operator for attribute comparison
    bool operator==(const DataPoint& other) const {
        return attributes == other.attributes;
    }

	// Overload equality operator for attribute comparison
	bool operator!=(const DataPoint& other) const {
		return attributes != other.attributes;
	}
};

// Function to select a random index
template<typename Iter>
size_t select_randomly(Iter start, Iter end) {
	static std::random_device rd; // Initialize random device
	static std::mt19937 gen(rd()); // Initialize Mersenne Twister pseudo-random number generator
	std::uniform_int_distribution<size_t> dis(0, std::distance(start, end) - 1); // Initialize uniform distribution for selecting random index

	return dis(gen);
}

double calculateSquaredEuclideanDistance(const DataPoint& point1, const DataPoint& point2) {
	double sum = 0.0;
	for (size_t i = 0; i < point1.attributes.size(); ++i) {
		sum += std::pow(point1.attributes[i] - point2.attributes[i], 2);
	}
	return sqrt(sum);
}

double calculateEuclideanDistance(const DataPoint& point1, const DataPoint& point2) {
	double sum = 0.0;
	for (size_t i = 0; i < point1.attributes.size(); ++i) {
		sum += std::pow(point1.attributes[i] - point2.attributes[i], 2);
	}
	return sum;
}

// Function for general file error handling
void handleFileError(const std::string& filename) {
	throw std::runtime_error("Error: Unable to open file '" + filename + "'");
}

// Function to read data points from a file
std::vector<DataPoint> readDataPoints(const std::string& filename) {
	// Open the file for reading, if the file failed to open, handle the error
	std::ifstream file(filename);
	if (!file.is_open()) {
		handleFileError(filename);
	}

	// Create a vector to store the data points
	std::vector<DataPoint> dataPoints;
	// Create a string to store each line of the file
	std::string line;

	// Read each line of the file
	while (std::getline(file, line)) {
		// Create a string stream to parse the line
		std::istringstream iss(line);
		// Create a vector to store the attributes of the data point
		std::vector<int> attributes;

		// Parse each value from the line and store it in the attributes vector
		int value;
		while (iss >> value) {
			attributes.push_back(value);
		}

		// Add the attributes to the data points vector
		dataPoints.emplace_back(attributes);
	}

	file.close();
	// Return the vector of data points
	return dataPoints;
}

// Function to get the number of dimensions in the data
int getNumDimensions(const std::string& filename) {
	// Open the file for reading, if the file failed to open, handle the error
	std::ifstream file(filename);
	if (!file.is_open()) {
		handleFileError(filename);
	}

	// Read the first line of the file
	std::string line;
	std::getline(file, line);

	// Check if the line is empty
	if (line.empty()) {
		// Print an error message and exit if the file is empty
		std::cerr << "Error: File '" << filename << "' is empty\n";
		std::exit(EXIT_FAILURE);
	}

	// Count the number of integers in the line
	std::istringstream iss(line);
	int dimensions = std::distance(std::istream_iterator<int>(iss), std::istream_iterator<int>());

	file.close();

	return dimensions;
}

std::vector<DataPoint> generateRandomCentroids(int numCentroids, const std::vector<DataPoint>& dataPoints) {
	std::srand(static_cast<unsigned int>(std::time(nullptr)));
	std::vector<DataPoint> dataPointsShuffled(dataPoints.begin(), dataPoints.end());
	std::random_shuffle(dataPointsShuffled.begin(), dataPointsShuffled.end());

	std::vector<DataPoint> centroids(numCentroids);
	for (int i = 0; i < numCentroids; ++i) {
		centroids[i] = dataPointsShuffled[i];
	}
	return centroids;
}

void writeCentroidsToFile(const std::string& filename, const std::vector<DataPoint>& centroids) {
	std::ofstream centroidFile(filename);
	if (!centroidFile.is_open()) {
		handleFileError(filename);
	}

	for (const DataPoint& centroid : centroids) {
		for (double attribute : centroid.attributes) {
			centroidFile << attribute << SEPARATOR;
		}
		centroidFile << "\n";
	}

	centroidFile.close();
	std::cout << "Centroid file created successfully: " << filename << std::endl;
}

//HUOM! numClusters + 1, tämä saattaa nyt aiheuttaa ongelmia
std::vector<int> generateRandomPartitions(int numDataPoints, int numClusters) {
	std::srand(static_cast<unsigned int>(std::time(nullptr)));
	std::vector<int> partitions(numDataPoints);
	for (int i = 0; i < numDataPoints; ++i) {
		partitions[i] = std::rand() % numClusters;
	}
	return partitions;
}

// Function to write partition to a text file
void writePartitionToFile(const std::vector<int>& partition, const std::string& fileName) {
	// Open the file for reading, if the file failed to open, handle the error
	std::ofstream outFile(fileName);
	if (!outFile.is_open()) {
		handleFileError(fileName);
	}

	// Write each data point and its corresponding cluster to the file
	for (size_t i = 0; i < partition.size(); ++i) {
		outFile << "Data Point " << i << ": Cluster " << partition[i] << "\n";
	}

	outFile.close();

	std::cout << "Optimal partition written to OptimalPartition.txt\n";
}

// Function to calculate SSE (Sum of Squared Errors) of a random centroid and all datapoints
double calculateRandomSSE(const std::vector<DataPoint>& dataPoints, const std::vector<DataPoint>& centroids) {
	
	// Select a random centroid
	int randomCentroidIndex = select_randomly(centroids.begin(), centroids.end());
	const DataPoint& chosenCentroid = centroids[randomCentroidIndex];

	double sse = 0.0;

	// Calculate SSE by summing squared distances between data points and the chosen centroid
	for (const DataPoint& dataPoint : dataPoints) {
		sse += calculateEuclideanDistance(dataPoint, chosenCentroid);
	}

	return sse;  // Return the calculated SSE
}

/*
* this is not used atm
* can be used to calcule SSE inside a cluster
double calculateSSE(const std::vector<DataPoint>& dataPoints, const std::vector<DataPoint>& centroids) {
	double sse = 0.0;

	for (const DataPoint& dataPoint : dataPoints) {
		double currentDistance = std::numeric_limits<int>::max();

		for (const DataPoint& centroid : centroids) {
			double newDistance = calculateEuclideanDistance(dataPoint, centroid);
			if (newDistance < currentDistance) {
				currentDistance = newDistance;
			}
		}

		sse += currentDistance;
	}

	return sse;
}*/

// Function to calculate total sum of squared errors (SSE)
double calculateSSE(const std::vector<DataPoint>& dataPoints, const std::vector<DataPoint>& centroids, const std::vector<int>& partition) {
	if (dataPoints.size() != partition.size()) {
		std::cerr << "Error: Data points and partition size mismatch\n";
		std::exit(EXIT_FAILURE);
	}

	double sse = 0.0;

	for (int i = 0; i < partition.size(); ++i) {
		// Find the index of the centroid assigned to the current data point
		int assignedCentroidIndex = partition[i];

		// Ensure the assigned centroid index is valid
		if (assignedCentroidIndex >= 0 && assignedCentroidIndex < static_cast<int>(centroids.size())) {
			// Calculate the squared distance between the data point and its assigned centroid
			sse += calculateEuclideanDistance(dataPoints[i], centroids[assignedCentroidIndex]);
		}
		else {
			std::cerr << "Error: Invalid centroid index in partition\n";
			std::exit(EXIT_FAILURE);
		}
	}

	return sse;
}

// Function to calculate pairwise distances and report average distance
void calculateAndReportAverageDistance(const std::vector<DataPoint>& dataPoints) {
	double totalDistance = 0.0;
	int pairwiseCount = 0;

	// Iterate through each pair of data points
	for (size_t i = 0; i < dataPoints.size(); ++i) {
		for (size_t j = i + 1; j < dataPoints.size(); ++j) {
			// Calculate distance between data points i and j
			totalDistance += calculateEuclideanDistance(dataPoints[i], dataPoints[j]);
			pairwiseCount++;
		}
	}

	if (pairwiseCount > 0) {
		double averageDistance = totalDistance / pairwiseCount;
		std::cout << "Average pairwise distance: " << averageDistance << std::endl;
	}
	else {
		std::cerr << "Error: No pairwise distances calculated (not enough data points)\n";
	}
}

// Function to find the nearest neighbor of a data point within a set of data
DataPoint findNearestNeighbor(DataPoint& queryPoint, const std::vector<DataPoint>& targetPoints) {
	// Throw an error if the set of data points is empty
	if (targetPoints.empty()) {
		throw std::runtime_error("Error: Cannot find nearest neighbor in an empty set of data");
	}

	// Initialize variables to store the minimum distance and the nearest neighbor
	double minDistance = std::numeric_limits<double>::max();//queryPoint.minDistance != -1 ? queryPoint.minDistance : -1;
	//HUOM!
	// EDELLINEN ALUSTETTIIN AINA MAX, JOTEN SEN TAKIA SE EI VOINUT VALITA, 
	// ETTÄ JATKETAAN SAMALLA CENTROIDILLA
	// onko edellinen itseasiassa bugi vai ei?
	DataPoint nearestNeighbor = queryPoint;

	// Iterate over each data point in the set
	for (const DataPoint& dataPoint : targetPoints) {
		// Compare the attributes of the query point and the current data point
		if (queryPoint != dataPoint) {
			// Calculate the distance between the query point and the current data point
			double distance = calculateEuclideanDistance(queryPoint, dataPoint);
			// Update the minimum distance and the nearest neighbor if the current data point is closer
			if (distance < minDistance || minDistance == -1) {
				minDistance = distance;
				nearestNeighbor = dataPoint;
			}
		}
	}

	// Return the nearest neighbor to the query point
	return nearestNeighbor;
}

// Function to find the nearest centroid of a data point
DataPoint findNearestCentroid(DataPoint& queryPoint, const std::vector<DataPoint>& clusterPoints) {
	// Throw an error if the set of data points is empty
	if (clusterPoints.empty()) {
		throw std::runtime_error("Error: Cannot find nearest centroid in an empty set of data");
	}

	int nearestCentroid = -1;

	std::vector<double> distances(clusterPoints.size());

	// Iterate over each data point in the set
	for (size_t i = 0; i < clusterPoints.size(); ++i) {
		// Calculate the distance between the query point and the current data point
		distances[i] = calculateEuclideanDistance(queryPoint, clusterPoints[i]);	
	}

	double minDistance = distances.front();

	for(int i = 0; i < clusterPoints.size(); ++i){
		if (distances[i] <= minDistance) {
			minDistance = distances[i];
			nearestCentroid = i;
		}
	}

	queryPoint.minDistance = minDistance;

	// Return the nearest neighbor to the query point
	return clusterPoints[nearestCentroid];
}

// Function for optimal partitioning
std::vector<int> optimalPartition(std::vector<DataPoint>& dataPoints, const std::vector<DataPoint>& centroids) {
	// Check if either dataPoints or centroids are empty
	if (dataPoints.empty() || centroids.empty()) {
		std::cerr << "Error: Cannot perform optimal partition with empty data or centroids\n";
		std::exit(EXIT_FAILURE);
	}

	// Create a vector to store the partition for each data point, initialized to -1
	std::vector<int> partition(dataPoints.size(), -1);

	// Iterate through each data point to find its nearest centroid
	for (size_t i = 0; i < dataPoints.size(); ++i) {
		DataPoint& dataPoint = dataPoints[i];
		DataPoint nearestCentroid;
		
		//debug helper
		//std::cout << "kierros: " << i << std::endl;

		if (std::find(centroids.begin(), centroids.end(), dataPoint) != centroids.end()) {
			
			nearestCentroid = dataPoint;
		}
		else {
			// Find the nearest centroid for the current data point
			nearestCentroid = findNearestCentroid(dataPoint, centroids);			
		}

		// Find the index of the nearest centroid in the centroids vector
		auto it = std::find(centroids.begin(), centroids.end(), nearestCentroid);
		if (it != centroids.end()) {
			// Update the partition with the index of the nearest centroid
			int centroidIndex = std::distance(centroids.begin(), it);
			partition[i] = centroidIndex;
		}
	}

	// Return the partition vector
	return partition;
}

// Calculate the centroid of a set of data points
DataPoint calculateCentroid(const std::vector<DataPoint>& dataPoints) {
	// Check if the set of data points is empty
	
	//TODO
	//KOODI KAATUU NYT TÄHÄN
	//TÄNNE SYÖTETÄÄN TYHJÄ VECTORI
	if (dataPoints.size() == 0) {
		throw std::runtime_error("Cannot calculate centroid for an empty set of data points");
	}

	DataPoint centroid;
	// Get the number of dimensions
	size_t numDimensions = dataPoints.front().attributes.size();

	// Loop through each dimension
	for (size_t dim = 0; dim < numDimensions; ++dim) {
		double sum = 0.0;

		// Calculate the sum of all data points in the current dimension
		for (const DataPoint& dataPoint : dataPoints) {
			sum += dataPoint.attributes[dim];
		}

		// Calculate the average for the current dimension and add it to the centroid
		centroid.attributes.push_back(sum / dataPoints.size());
	}

	// Return the centroid
	return centroid;
}

// Implementation of the writeCentroidsToFile function
void writeCentroidsToFile(const std::vector<DataPoint>& centroids, const std::string& fileName) {
	// Open the file for reading, if the file failed to open, handle the error
	std::ofstream outFile(fileName);
	if (!outFile.is_open()) {
		handleFileError(fileName);
	}

	// Iterate through each centroid
	for (size_t i = 0; i < centroids.size(); ++i) {
		// Write the centroid label to the file
		outFile << "Centroid for Cluster " << i + 1 << ":\n";
		// Iterate through each dimension of the centroid and write it to the file
		for (size_t dim = 0; dim < centroids[i].attributes.size(); ++dim) {
			outFile << centroids[i].attributes[dim] << " ";
		}
		outFile << "\n";
	}
	outFile.close();

	std::cout << "Constructed centroids written to ConstructedCentroids.txt\n";
}

// Function to perform the centroid step in k-means
std::vector<DataPoint> kMeansCentroidStep(const std::vector<DataPoint>& dataPoints, const std::vector<int>& partition, int numClusters) {
	std::vector<DataPoint> newCentroids(numClusters);

	// Group data points by cluster
	std::vector<std::vector<DataPoint>> clusters(numClusters);

	// Initialize clusters with empty vectors
	for (int i = 0; i < numClusters; ++i) {
		clusters[i] = std::vector<DataPoint>();
	}

	//HUOM
	//Koodi kaatuu nyt tähän, sillä parition vectorissa löytyy rivejä, joilla on arvona -1
	//
	// BUG1 (korjattu)
	// partitionit alustetaan arvolle -1
	//eli nyt ongelmana on se, että data pointit, jotka valitaan centroideiksi jäävät arvoille -1 (korjattu)
	//
	// BUG2
	//ongelma saattaa olla nyt siinä, että ensin valitaan centroidit randomisti -> minDistance = 0 -> minDistance != -1 -> nearestNeighbor ei päivity -> jää arvoon -1

	for (int i = 0; i < dataPoints.size(); ++i) {
		int clusterLabel = partition[i];
		clusters[clusterLabel].push_back(dataPoints[i]);
	}

	//BUG
	//tänne tulee nyt clustereita joissa on 0 datapointtia, ja homma räjähtää
	// Calculate centroid for each cluster using the calculateCentroid function
	for (int clusterLabel = 0; clusterLabel < numClusters; ++clusterLabel) {
		newCentroids[clusterLabel] = calculateCentroid(clusters[clusterLabel]);
	}

	return newCentroids;
}

void findNearestNeighborExample(const std::vector<DataPoint>& dataPoints) {
	// Select a random data point from the dataset
	DataPoint firstDataPoint = dataPoints[select_randomly(dataPoints.begin(), dataPoints.end())];

	// Find the nearest neighbor of the selected data point
	DataPoint nearestNeighbor = findNearestNeighbor(firstDataPoint, dataPoints);

	// Print the original data point and the nearest neighbor
	std::cout << "Nearest neighbor of the first data point:\n";
	std::cout << "Original Data Point: ";
	for (int value : firstDataPoint.attributes) {
		std::cout << value << " ";
	}
	std::cout << "\nNearest Neighbor: ";
	for (int value : nearestNeighbor.attributes) {
		std::cout << value << " ";
	}
	std::cout << "\n";
}

double runKMeans(std::vector<DataPoint>& dataPoints, int iterations, std::vector<DataPoint>& centroids) {
	std::vector<double> bestCentroids;
	double bestSse = std::numeric_limits<double>::max();
	int stopCounter = 0;
	double previousSSE = std::numeric_limits<double>::max();
	std::vector<int> previousPartition(dataPoints.size(), -1);
	std::vector<int> activeClusters;

	for (int iteration = 0; iteration < iterations; ++iteration) {
		// Calculate new centroids and update the partition
		std::vector<int> newPartition = optimalPartition(dataPoints, centroids);
		activeClusters.clear();

		centroids = kMeansCentroidStep(dataPoints, newPartition, NUM_CENTROIDS);
		//Activity
		for (int i = 0; i < newPartition.size(); ++i) {

			if (newPartition[i] != previousPartition[i]) {
				if(std::find(activeClusters.begin(), activeClusters.end(), newPartition[i]) == activeClusters.end())
				{
					activeClusters.push_back(newPartition[i]);
				}
				if (previousPartition[i] != -1 && std::find(activeClusters.begin(), activeClusters.end(), previousPartition[i]) == activeClusters.end())
				{
					activeClusters.push_back(previousPartition[i]);
				}
			}
		}

		// Calculate and report sum-of-squared errors
		double sse = calculateSSE(dataPoints, centroids, newPartition);
		std::cout << "Total SSE after iteration " << iteration + 1 << ": " << sse << "and activity: " << activeClusters.size() << std::endl;

		if (sse < bestSse) {
			bestSse = sse;
		}
		else if (sse == previousSSE) {
			stopCounter++;
		}

		// For now, we use stopCounter
		// Optionally, check for convergence or other stopping criteria		
		if (stopCounter == 3) {
			break;
		}

		previousSSE = sse;
		previousPartition = newPartition;
	}

	return bestSse;
}

double randomSwap(std::vector<DataPoint>& dataPoints, std::vector<DataPoint>& centroids) {
	int swaps = 45;
	int iterations = 2;

	double bestSse = std::numeric_limits<double>::max();
	DataPoint oldCentroid;

	for (int i = 0; i < swaps; ++i) {

		//Valitse centroid
		int randomCentroid = select_randomly(centroids.begin(), centroids.end());
		oldCentroid = centroids[randomCentroid];

		//Valitse datapiste
		int randomDataPoint = select_randomly(dataPoints.begin(), dataPoints.end());

		//Suorita vaihto
		centroids[randomCentroid] = dataPoints[randomDataPoint];


		//Run k-means twice	
		double sse = runKMeans(dataPoints, 2, centroids);

		//If SSE improves, we keep the change
		//if not, then we reverse the swap
		if (sse < bestSse) {
			bestSse = sse;
		}
		else {
			centroids[randomCentroid] = oldCentroid;
		}
	}

	return bestSse;
}

int main() {
	// Get the number of dimensions in the data
	int numDimensions = getNumDimensions(DATA_FILENAME);

	if (numDimensions != -1) {
		std::cout << "Number of dimensions in the data: " << numDimensions << std::endl;

		// Read data points
		std::vector<DataPoint> dataPoints = readDataPoints(DATA_FILENAME);

		// Generate and write centroids
		std::vector<DataPoint> centroids = generateRandomCentroids(NUM_CENTROIDS, dataPoints);
		std::vector<DataPoint> ogCentroids(centroids.begin(), centroids.end());

		writeCentroidsToFile(centroids, CENTROID_FILENAME);

		// Generate and write random partitions
		writePartitionToFile(generateRandomPartitions(dataPoints.size(), NUM_CENTROIDS), PARTITION_FILENAME);

		// Calculate and report sum-of-squared errors
		double sse = calculateRandomSSE(dataPoints, centroids);
		std::cout << "(Start) Sum-of-Squared Errors (SSE): " << sse << std::endl;

		// Calculate and report average pairwise distance
		calculateAndReportAverageDistance(dataPoints);


		// Example: Find the nearest neighbor of a random data point in the original dataset
		// Disabled, as nearestNeighbor updates minDistance --> might cause problems
		//findNearestNeighborExample(dataPoints);

		// Initial partition and SSE using random centroids
		std::vector<int> initialPartition = optimalPartition(dataPoints, centroids);
		double initialSSE = calculateSSE(dataPoints, centroids, initialPartition);

		// Write the initial partition to a text file
		writePartitionToFile(initialPartition, OUTPUTS_FOLDER + "OptimalPartition.txt");

		// Print initial SSE
		std::cout << "Initial Total Sum-of-Squared Errors (SSE): " << initialSSE << std::endl;

		double bestSse1 = initialSSE;
		int stopCounter = 0;

		//run just k-means
		//runKMeans(dataPoints, MAX_ITERATIONS, centroids);
		
		//Repeated k-means
		for (int repeat = 0; repeat < MAX_REPEATS; ++repeat) {
			std::cout << "round: " << repeat << std::endl;

			// New centroids
			centroids = generateRandomCentroids(NUM_CENTROIDS, dataPoints);

			double newSse = runKMeans(dataPoints, MAX_ITERATIONS, centroids);	

			if (newSse < bestSse1) {
				bestSse1 = newSse;
			}
		}

		// Write the constructed centroids to a text file (optional)
		writeCentroidsToFile(centroids, OUTPUTS_FOLDER + "ConstructedCentroids.txt");

		//Random swap
		double bestSse2 = randomSwap(dataPoints, ogCentroids);

		std::cout << "(Naive)Best Sum-of-Squared Errors (SSE): " << bestSse1 << std::endl;
		std::cout << "(RS)Best Sum-of-Squared Errors (SSE): " << bestSse2 << std::endl;

		return 0;
	}
}