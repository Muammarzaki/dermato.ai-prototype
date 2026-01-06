package com.github.dermatoai.data.db

import androidx.room.Database
import androidx.room.RoomDatabase
import com.github.dermatoai.data.db.dao.PredictionRecordDao
import com.github.dermatoai.data.db.entity.PredictionRecordEntity

@Database(
    entities = [
        PredictionRecordEntity::class
    ],
    version = 1,
    exportSchema = false
)
abstract class AppDatabase : RoomDatabase() {
    abstract fun predictionDao(): PredictionRecordDao
}
