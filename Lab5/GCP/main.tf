provider "google" {
    project     = "mlops-project-476416"
    region      = "us-central1"
    zone        = "us-central1-a"
    credentials = "mlops-project-476416-56b7ec3ce8c1.json"
}

# Create bucket
resource "google_storage_bucket" "my_bucket" {
    name          = "bucket-12345-terraform"
    location      = "us-central1"
    force_destroy = true
}

# Upload image
resource "google_storage_bucket_object" "my_image" {
    name   = "image.jpeg"
    bucket = google_storage_bucket.my_bucket.name
    source = "image.jpeg"
}

# Make image public
resource "google_storage_bucket_iam_member" "public" {
    bucket = google_storage_bucket.my_bucket.name
    role   = "roles/storage.objectViewer"
    member = "allUsers"
}

# Create VM
resource "google_compute_instance" "vm" {
    name         = "my-vm"
    machine_type = "e2-micro"

    boot_disk {
        initialize_params {
            image = "debian-cloud/debian-11"
        }
    }

    network_interface {
        network = "default"
        access_config {}
    }

    tags = ["http-server"]

    metadata_startup_script = <<-EOF
        #!/bin/bash
        apt-get update
        apt-get install -y nginx
        cat > /var/www/html/index.html <<HTML
        <html>
        <body style="text-align:center; padding:50px;">
            <h1>My Image</h1>
            <img src="https://storage.googleapis.com/test-bucket-terraform/image.jpeg" width="600">
        </body>
        </html>
HTML
        systemctl restart nginx
        EOF
}

# Allow HTTP
resource "google_compute_firewall" "http" {
    name    = "allow-http"
    network = "default"
    allow {
        protocol = "tcp"
        ports    = ["80"]
    }
    source_ranges = ["0.0.0.0/0"]
    target_tags   = ["http-server"]
}

# Show VM IP
output "visit_here" {
    value = "http://${google_compute_instance.vm.network_interface[0].access_config[0].nat_ip}"
}