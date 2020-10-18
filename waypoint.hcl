project = "fastmodel-stater"

app "fastmodel-stater" {
  labels = {
    "service" = "fastmodel-stater",
    "env" = "dev"
  }

  build {
    use "docker" {}
  }

  deploy { 
    use "docker" {
        service_port = 8080
    }
  }
}