// Each angle assigned a binary value (powers of 2)
const angles = {
    "L_Elbow": 1,    // 2^0
    "L_Hip": 2,      // 2^1
    "R_Elbow": 4,    // 2^2
    "R_Hip": 8,      // 2^3
    "L_Arm": 16,     // 2^4
    "R_Arm": 32,     // 2^5
    "L_Back": 64,    // 2^6
    "R_Back": 128,   // 2^7
    "L_Knee": 256,   // 2^8
    "R_Knee": 512    // 2^9
  };

  const container = document.getElementById('checkbox-list');

  // Create checkboxes dynamically
  for (const key in angles) {
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.id = key;
    checkbox.name = 'angle';
    checkbox.value = key;

    const label = document.createElement('label');
    label.htmlFor = key;
    label.textContent = key.replace("_", " ");

    container.appendChild(checkbox);
    container.appendChild(label);
    container.appendChild(document.createElement('br'));
  }

  // Calculate binary sum when button clicked
  function calculateBinarySum() {
    const selectedCheckboxes = document.querySelectorAll('input[name="angle"]:checked');
    let totalValue = 0;

    selectedCheckboxes.forEach(checkbox => {
      totalValue += angles[checkbox.value];
    });
    return totalValue;
    // document.getElementById('result1').textContent = 
    //   "Binary Encoded Sum: " + totalValue;
  }