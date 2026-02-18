// src/lib.rs

// A simple public function
pub fn greet() {
    println!("Hello from the library!");
}

// Defining a private internal function
fn internal_logic() {
    println!("This can't be seen outside this crate.");
}