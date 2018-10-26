# Load MASS biopsy data set.
library(MASS)
library(neuralnet)

# Copy data into personal data set
my.biopsy <- biopsy

# my.biopsy$class has a string character type.  These need to be 0-1.
# Create 0-1 variable for $class ("benign","malignant")
my.biopsy$binary <- ifelse(my.biopsy$class=="benign",0,1)

# Is there missing data?
# Yes, V6 = bare nuclei missing 16 values.  See documentation.
apply(my.biopsy, 2, function(x) sum(is.na(x)))

# Omit the data
my.biopsy2 <- na.omit(my.biopsy)

# Make all non-ID numerical values from 0-1.

normalize <- function(x)
{
  return((x- min(x)) /(max(x)-min(x)))
}

# To get a vector, use apply instead of lapply.
my.data <- as.data.frame(lapply(my.biopsy2[2:10], normalize))

# Bind numeric with binary vector.
my.data <- cbind(my.data, my.biopsy2$binary)

index <- sample(1:nrow(my.data),round(0.75*nrow(my.data)))
train <- my.data[index,]
test <- my.data[-index,]

# Run the neural net.  This example is just on V1, thickness.
nn <- neuralnet(V9 ~ V1,data=train,hidden=c(4,1),linear.output=T)

plot(nn)
nn$result.matrix

# Run on the text data.
pr.nn <- compute(nn,test[,1])
