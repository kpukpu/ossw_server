import React from "react";
import styles from "./Creating.module.css";
import creating from "../../assets/images/creating.png";

function Creating() {
  return (
    <div className={styles.container}>
      <img src={creating} alt="creating" />
      <p className={styles.textAnimation}>기사 생성 중...</p>
    </div>
  );
}

export default Creating;
