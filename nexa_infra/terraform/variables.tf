variable "aws_region" {
  description = "AWS Region to deploy to"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "gpu_instance_type" {
  description = "EC2 Instance type for GPU workers"
  type        = string
  default     = "g5.2xlarge"
}

variable "node_count" {
  description = "Number of GPU worker nodes"
  type        = number
  default     = 1
}

