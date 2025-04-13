// static/js/tooltips.js
function toggleTooltip(elem) {
    var tooltip = elem.nextElementSibling;
    if (tooltip.style.display === "none" || tooltip.style.display === "") {
      tooltip.style.display = "block";
    } else {
      tooltip.style.display = "none";
    }
  }
  
  // Hide any tooltip when clicking outside
  document.addEventListener('click', function(event) {
    var tooltips = document.querySelectorAll('.tooltip');
    tooltips.forEach(function(tooltip) {
      if (!tooltip.parentElement.contains(event.target)) {
        tooltip.style.display = "none";
      }
    });
  });
  