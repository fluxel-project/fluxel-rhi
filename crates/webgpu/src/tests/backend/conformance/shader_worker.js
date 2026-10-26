export function nextTask() {
  return new Promise(resolve => setTimeout(resolve, 0));
}
