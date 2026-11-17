
document.addEventListener("DOMContentLoaded", function () {

                  // Smooth scrolling for navigation links
                  document.querySelectorAll('a[href^="#"]').forEach(function (link) {
                      link.addEventListener("click", function (event) {
                          const target = document.querySelector(this.getAttribute("href"));
              
                          if (target) {
                              event.preventDefault();
                              target.scrollIntoView({
                                  behavior: "smooth"
                              });
                          }
                      });
                  });
              
                  // Scroll to prediction result after form submission
                  const result = document.getElementById("result");
              
                  if (result) {
                      result.scrollIntoView({
                          behavior: "smooth",
                          block: "center"
                      });
                  }
              
                  // Prevent invalid or negative vehicle values
                  const form = document.querySelector(".prediction-form");
              
                  if (form) {
                      form.addEventListener("submit", function (event) {
              
                          const numericFields = form.querySelectorAll(
                              'input[type="number"]'
                          );
              
                          for (const field of numericFields) {
                              if (field.value.trim() === "" ||
                                  !Number.isFinite(Number(field.value)) ||
                                  Number(field.value) < 0) {
              
                                  event.preventDefault();
                                  alert("Please enter valid positive vehicle details.");
                                  field.focus();
                                  return;
                              }
                          }
                      });
                  }
              
              });
              