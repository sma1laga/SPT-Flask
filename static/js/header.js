// Wait for the page to load
document.addEventListener("DOMContentLoaded", function() {
  const darkModeToggle = document.getElementById("dark-mode-toggle");
  if (darkModeToggle) {  
    darkModeToggle.addEventListener("click", function() {
      document.body.classList.toggle("dark-mode");
      // Save the preference in local storage
      if (document.body.classList.contains("dark-mode")) {
        localStorage.setItem("darkMode", "true");
        darkModeToggle.textContent = "Light Mode";
      } else {
        localStorage.setItem("darkMode", "false");
        darkModeToggle.textContent = "Dark Mode";
      }
    });
  
    // On page load, check local storage for dark mode preference
    if (localStorage.getItem("darkMode") === "true") {
      document.body.classList.add("dark-mode");
      darkModeToggle.textContent = "Light Mode";
    }
  }

  // Sidebar toggle for mobile
  const sidebarToggle = document.getElementById("sidebar-toggle");
  const sidebar = document.querySelector(".sidebar");
  if (sidebarToggle && sidebar) {
    sidebarToggle.addEventListener("click", function() {
      sidebar.classList.toggle("active");
    });
  }
});
